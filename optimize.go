package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
)

// 优化请求结构
type OptimizationRequest struct {
	PipelineID      string             `json:"pipelineId"`
	MainTable       TableData          `json:"mainTable"`
	AuxiliaryTables []TableData        `json:"auxiliaryTables"`
	ProcessingSteps []StepConfig       `json:"processingSteps"`
	OutputConfig    OutputConfig       `json:"outputConfig"`
	Optimization    OptimizationConfig `json:"optimization"`
}

// 优化配置
type OptimizationConfig struct {
	Scenario         string                 `json:"scenario"`
	UploadedFileName string                 `json:"uploadedFileName"`
	DataColumns      []string               `json:"dataColumns"`
	Parameters       map[string]interface{} `json:"parameters"`
}

// 优化结果
type OptimizationResult struct {
	ExecutionID     string                 `json:"executionId"`
	Scenario        string                 `json:"scenario"`
	OptimalValues   []VariableResult       `json:"optimalValues"`
	ObjectiveValues []float64              `json:"objectiveValues"`
	Constraints     []ConstraintResult     `json:"constraints"`
	ExecutionTime   string                 `json:"executionTime"`
	Charts          map[string]interface{} `json:"charts"`
	ResultFile      string                 `json:"resultFile"`
}

type VariableResult struct {
	Name  string  `json:"name"`
	Value float64 `json:"value"`
}

type ConstraintResult struct {
	Name      string  `json:"name"`
	Value     float64 `json:"value"`
	Satisfied bool    `json:"satisfied"`
}

// 注册优化相关路由
func RegisterOptimizationRoutes(r *gin.Engine) {
	r.POST("/api/optimization/execute-pipeline", authMiddleware, executeOptimizationPipeline)
	r.GET("/api/optimization/results/:id", authMiddleware, getOptimizationResult)
	r.GET("/api/optimization/executions", authMiddleware, getOptimizationExecutions)
}

// 执行优化流水线
func executeOptimizationPipeline(c *gin.Context) {
	logrus.Info("=== executeOptimizationPipeline ===")

	// 获取当前用户
	user, ok := getCurrentUser(c)
	if !ok {
		c.JSON(401, gin.H{"status": "error", "message": "未授权访问"})
		return
	}

	var req OptimizationRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		logrus.Error("参数绑定错误:", err)
		c.JSON(400, gin.H{"status": "error", "message": "参数错误: " + err.Error()})
		return
	}

	// 生成唯一的执行ID
	executionID := req.PipelineID
	if executionID == "" {
		executionID = fmt.Sprintf("optim_%s", uuid.New().String()[:8])
	}

	// 创建优化工作目录
	baseDir := "./optimizations"
	workDir := filepath.Join(baseDir, executionID)

	// 创建目录
	if err := os.MkdirAll(workDir, 0755); err != nil {
		logrus.Error("创建工作目录失败:", err)
		c.JSON(500, gin.H{"status": "error", "message": "创建工作目录失败"})
		return
	}

	// 保存配置文件
	configFile := filepath.Join(workDir, "config.json")
	configData, err := json.MarshalIndent(req, "", "  ")
	if err != nil {
		logrus.Error("序列化配置失败:", err)
		c.JSON(500, gin.H{"status": "error", "message": "保存配置失败"})
		return
	}

	if err := ioutil.WriteFile(configFile, configData, 0644); err != nil {
		logrus.Error("写入配置文件失败:", err)
		c.JSON(500, gin.H{"status": "error", "message": "写入配置文件失败"})
		return
	}

	// 创建执行记录
	record := PipelineExecutionRecord{
		ID:           executionID,
		PipelineID:   executionID,
		PipelineName: fmt.Sprintf("Optimization_%s", req.Optimization.Scenario),
		Username:     user.Username,
		Status:       "processing",
		RequestData:  string(configData),
		CreatedAt:    time.Now(),
	}

	if err := db.Create(&record).Error; err != nil {
		logrus.Error("创建执行记录失败:", err)
		c.JSON(500, gin.H{"status": "error", "message": "创建执行记录失败"})
		return
	}

	// 执行Python优化脚本
	startTime := time.Now()
	resultFile := filepath.Join(workDir, "result.json")

	// 构建Python命令
	cmd := exec.Command("python",
		"-m", "optimization.run",
		"--config", configFile,
		"--output", resultFile,
		"--scenario", req.Optimization.Scenario)

	// 设置工作目录
	cmd.Dir = workDir

	// 捕获输出
	output, err := cmd.CombinedOutput()
	executionTime := time.Since(startTime)

	if err != nil {
		logrus.Error("Python优化脚本执行失败:", err, "输出:", string(output))

		// 更新记录状态为失败
		db.Model(&record).Updates(map[string]interface{}{
			"status":         "failed",
			"error_message":  fmt.Sprintf("优化执行失败: %s, 输出: %s", err, string(output)),
			"execution_time": fmt.Sprintf("%.2fs", executionTime.Seconds()),
			"completed_at":   time.Now(),
		})

		c.JSON(500, gin.H{
			"status":      "error",
			"message":     "优化执行失败",
			"error":       err.Error(),
			"output":      string(output),
			"executionId": executionID,
		})
		return
	}

	// 读取结果文件
	resultData, err := ioutil.ReadFile(resultFile)
	if err != nil {
		logrus.Error("读取结果文件失败:", err)

		db.Model(&record).Updates(map[string]interface{}{
			"status":         "completed_with_warnings",
			"error_message":  "无法读取结果文件",
			"execution_time": fmt.Sprintf("%.2fs", executionTime.Seconds()),
			"completed_at":   time.Now(),
		})

		c.JSON(200, gin.H{
			"status":  "success",
			"message": "优化完成但结果文件读取失败",
			"data": gin.H{
				"executionId":   executionID,
				"executionTime": fmt.Sprintf("%.2fs", executionTime.Seconds()),
				"output":        string(output),
			},
		})
		return
	}

	// 解析结果
	var optimizationResult OptimizationResult
	if err := json.Unmarshal(resultData, &optimizationResult); err != nil {
		logrus.Error("解析结果失败:", err)
		optimizationResult = OptimizationResult{
			ExecutionID: executionID,
			Scenario:    req.Optimization.Scenario,
			ResultFile:  resultFile,
		}
	}

	// 更新记录
	resultJSON, _ := json.Marshal(optimizationResult)
	db.Model(&record).Updates(map[string]interface{}{
		"status":         "success",
		"step_results":   string(resultJSON),
		"final_output":   string(resultJSON),
		"execution_time": fmt.Sprintf("%.2fs", executionTime.Seconds()),
		"completed_at":   time.Now(),
	})

	// 返回结果
	c.JSON(200, gin.H{
		"status":      "success",
		"message":     "优化执行成功",
		"data":        optimizationResult,
		"executionId": executionID,
	})
}

// 获取优化结果
func getOptimizationResult(c *gin.Context) {
	executionID := c.Param("id")

	// 读取结果文件
	resultFile := filepath.Join("./optimizations", executionID, "result.json")
	resultData, err := ioutil.ReadFile(resultFile)

	if err != nil {
		// 从数据库获取记录
		var record PipelineExecutionRecord
		if err := db.Where("id = ?", executionID).First(&record).Error; err != nil {
			c.JSON(404, gin.H{"status": "error", "message": "结果不存在"})
			return
		}

		// 返回数据库中的结果
		var result OptimizationResult
		json.Unmarshal([]byte(record.FinalOutput), &result)

		c.JSON(200, gin.H{
			"status": "success",
			"data":   result,
		})
		return
	}

	// 解析并返回结果文件
	var result OptimizationResult
	json.Unmarshal(resultData, &result)

	c.JSON(200, gin.H{
		"status": "success",
		"data":   result,
	})
}

// 获取优化执行列表
func getOptimizationExecutions(c *gin.Context) {
	user, _ := getCurrentUser(c)

	var records []PipelineExecutionRecord
	db.Where("username = ? AND pipeline_name LIKE ?", user.Username, "Optimization_%").
		Order("created_at DESC").
		Find(&records)

	c.JSON(200, gin.H{
		"status": "success",
		"data":   records,
	})
}
