package main

import (
	"crypto/md5"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/xuri/excelize/v2"
	"gorm.io/gorm"
)

// 表数据结构
type TableData struct {
	ID          string                   `json:"id"`
	Name        string                   `json:"name"`
	FileName    string                   `json:"fileName"`
	Filehash    string                   `json:"filehash"`
	Type        string                   `json:"type"`
	Columns     []string                 `json:"columns"`
	PreviewData []map[string]interface{} `json:"previewData"`
	FullData    []map[string]interface{} `json:"fullData"`
}

// 步骤配置
type StepConfig struct {
	ID     string                 `json:"id"`
	Type   string                 `json:"type"`
	Config map[string]interface{} `json:"config"`
	Index  int                    `json:"index"`
}

// 输出配置
type OutputConfig struct {
	Format       string `json:"format"`
	IncludeSteps bool   `json:"includeSteps"`
}

// 流水线请求
type PipelineRequest struct {
	PipelineID      string       `json:"pipelineId"`
	MainTable       TableData    `json:"mainTable"`
	AuxiliaryTables []TableData  `json:"auxiliaryTables"`
	ProcessingSteps []StepConfig `json:"processingSteps"`
	OutputConfig    OutputConfig `json:"outputConfig"`
}

// 步骤执行结果
type StepResult struct {
	StepID        string                   `json:"stepId"`
	Success       bool                     `json:"success"`
	Message       string                   `json:"message"`
	OutputColumns []string                 `json:"outputColumns"`
	SampleData    []map[string]interface{} `json:"sampleData"`
	DataLength    int                      `json:"dataLength"`
	ExecutionTime string                   `json:"executionTime"`
}

// 流水线响应
type PipelineResponse struct {
	Success     bool         `json:"success"`
	Message     string       `json:"message"`
	TotalSteps  int          `json:"totalSteps"`
	FinalOutput *FinalOutput `json:"finalOutput"`
	StepResults []StepResult `json:"stepResults"`
}

type FinalOutput struct {
	Columns    []string `json:"columns"`
	DataLength int      `json:"dataLength"`
	OutputFile string   `json:"outputFile"`
}

// 保存配置请求
type SaveTablePiplineConfigRequest struct {
	ID            string       `json:"id"`
	Name          string       `json:"name"`
	PipelineSteps []StepConfig `json:"pipelineSteps"`
	Tables        []TableData  `json:"tables"`
	OutputConfig  OutputConfig `json:"outputConfig"`
}

// 配置存储
var savedConfigs = make(map[string]SaveTablePiplineConfigRequest)

type TablePipelineConfig struct {
	ID            string    `json:"id" gorm:"primaryKey"`
	Name          string    `json:"name"`
	PipelineSteps string    `json:"pipeline_steps" gorm:"type:json"` // 存储步骤配置
	Tables        string    `json:"tables" gorm:"type:json"`         // 存储表数据
	OutputConfig  string    `json:"output_config" gorm:"type:json"`  // 存储输出配置
	Username      string    `json:"username"`
	CreatedAt     time.Time `json:"created_at"`
	UpdatedAt     time.Time `json:"updated_at"`

	Deletedat gorm.DeletedAt `gorm:"index" json:"-"`
}

// 流水线执行记录表
type PipelineExecutionRecord struct {
	ID            string    `json:"id" gorm:"primaryKey"`
	PipelineID    string    `json:"pipeline_id"` // 关联的配置ID
	PipelineName  string    `json:"pipeline_name"`
	Username      string    `json:"username"`
	Status        string    `json:"status"`                        // processing, success, failed
	RequestData   string    `json:"request_data" gorm:"type:json"` // 完整的请求数据
	StepResults   string    `json:"step_results" gorm:"type:json"`
	FinalOutput   string    `json:"final_output" gorm:"type:json"`
	ErrorMessage  string    `json:"error_message,omitempty"`
	ExecutionTime string    `json:"execution_time,omitempty"`
	CreatedAt     time.Time `json:"created_at"`
	CompletedAt   time.Time `json:"completed_at,omitempty"`
}

func InitTableDatapiplineTables(db *gorm.DB) error {
	if err := db.AutoMigrate(
		&TablePipelineConfig{},
		&PipelineExecutionRecord{},
	); err != nil {
		return fmt.Errorf("自动迁移表格数据流水线表结构失败: %v", err)
	}

	logger.Info("表格数据流水线表结构初始化完成")
	return nil
}

func RegistertablePipleRoutes(r *gin.Engine) {
	r.POST("/api/tablepipeline/execute-pipeline", authMiddleware, executePipeline)
	r.GET("/api/tablepipeline/executions", authMiddleware, getPipelineExecutionRecords)   // 新增：获取执行记录列表
	r.GET("/api/tablepipeline/execution/:id", authMiddleware, getPipelineExecutionRecord) // 新增：获取执行记录详情
	r.POST("/api/tablepipeline/save-config", authMiddleware, saveTablepiplineConfig)
	r.GET("/api/tablepipeline/configs", authMiddleware, getTablepipelineConfigList)
	r.GET("/api/tablepipeline/config/:id", authMiddleware, getTablepiplineConfigDetail)
	r.DELETE("/api/tablepipeline/config/:id", authMiddleware, deleteTablePiplineConfig)
	r.POST("/api/tablepipeline/upload-file", authMiddleware, uploadTableFile) // 新增文件上传接口
	r.POST("/api/tablepipeline/load-from-url", authMiddleware, loadTableFromURL)
}

// 执行流水线
func executePipeline(c *gin.Context) {
	logger.Info("=== executePipeline ===")

	// 获取当前用户名
	user, ok := getCurrentUser(c)
	if !ok {
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "获取用户信息失败"})
		return
	}

	var req PipelineRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		logger.Error("参数绑定错误:", err)
		c.JSON(http.StatusBadRequest, gin.H{"status": "error", "message": "参数错误: " + err.Error()})
		return
	}

	logger.Info("=== PipelineRequest :", req)

	// 验证必要数据
	if req.MainTable.FileName == "" {
		c.JSON(http.StatusBadRequest, gin.H{"status": "error", "message": "主表文件不能为空"})
		return
	}

	if len(req.ProcessingSteps) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"status": "error", "message": "处理步骤不能为空"})
		return
	}

	// 序列化完整的请求数据
	requestDataJSON, err := json.Marshal(req)
	if err != nil {
		logger.Error("序列化请求数据失败:", err)
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "处理请求数据失败"})
		return
	}

	// 创建处理记录
	processRecord := PipelineExecutionRecord{
		ID:           fmt.Sprintf("pipeline_%s", time.Now().Format("20060102150405")),
		PipelineID:   req.PipelineID,
		PipelineName: fmt.Sprintf("Pipeline_%s", time.Now().Format("20060102150405")),
		Username:     user.Username,
		Status:       "processing",
		RequestData:  string(requestDataJSON),
		CreatedAt:    time.Now(),
	}

	if err := db.Create(&processRecord).Error; err != nil {
		logger.Error("创建处理记录失败:", err)
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "处理记录创建失败"})
		return
	}

	// 同步执行Python处理流水线（阻塞等待）
	startTime := time.Now()
	output, err := executeTablePipelinePythonProcess(processRecord.ID, req.MainTable, req.AuxiliaryTables, req.ProcessingSteps, req.OutputConfig)
	executionTime := time.Since(startTime)

	if err != nil {
		logger.Error("executeTablePipelinePythonProcess error detail: ", err, output)

		// 更新处理记录状态为失败
		db.Model(&processRecord).Updates(map[string]interface{}{
			"status":         "failed",
			"error_message":  fmt.Sprintf("任务执行失败: %s", err),
			"step_results":   output,
			"execution_time": fmt.Sprintf("%.2fs", executionTime.Seconds()),
			"completed_at":   time.Now(),
		})

		c.JSON(http.StatusInternalServerError, gin.H{
			"status":  "error",
			"message": fmt.Sprintf("任务执行失败: %s", err),
			"data":    output, // 返回完整的结构化错误信息
		})
		return
	}

	// 处理成功，更新记录状态
	db.Model(&processRecord).Updates(map[string]interface{}{
		"status":         "success",
		"step_results":   output,
		"execution_time": fmt.Sprintf("%.2fs", executionTime.Seconds()),
		"completed_at":   time.Now(),
	})

	// 解析输出结果返回给前端
	var resultData map[string]interface{}
	if err := json.Unmarshal([]byte(output), &resultData); err != nil {
		// 如果无法解析为JSON，直接返回原始输出
		resultData = map[string]interface{}{
			"output": output,
		}
	}

	// 存储最终输出
	if finalOutput, exists := resultData["output_file"]; exists {
		finalOutputJSON, _ := json.Marshal(finalOutput)
		db.Model(&processRecord).Update("output_file", string(finalOutputJSON))
	}

	c.JSON(http.StatusOK, gin.H{
		"status":  "success",
		"message": "处理链执行成功",
		"data":    resultData,
	})
}

func getPipelineExecutionRecords(c *gin.Context) {
	user, ok := getCurrentUser(c)
	if !ok {
		c.JSON(http.StatusUnauthorized, gin.H{
			"success": false,
			"message": "用户未认证",
		})
		return
	}

	var records []PipelineExecutionRecord
	if err := db.Where("username = ?", user.Username).Order("created_at DESC").Find(&records).Error; err != nil {
		logger.Errorf("查询执行记录失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"success": false,
			"message": "获取执行记录失败: " + err.Error(),
		})
		return
	}

	// 转换为前端需要的格式
	result := make([]map[string]interface{}, 0)
	for _, record := range records {
		// 解析请求数据获取基本信息
		var requestData PipelineRequest
		var mainTableName string
		var stepsCount int

		if err := json.Unmarshal([]byte(record.RequestData), &requestData); err == nil {
			mainTableName = requestData.MainTable.Name
			stepsCount = len(requestData.ProcessingSteps)
		} else {
			mainTableName = "未知表"
			stepsCount = 0
		}

		result = append(result, map[string]interface{}{
			"id":             record.ID,
			"pipeline_id":    record.PipelineID,
			"pipeline_name":  record.PipelineName,
			"main_table":     mainTableName,
			"steps_count":    stepsCount,
			"status":         record.Status,
			"execution_time": record.ExecutionTime,
			"created_at":     record.CreatedAt.Format("2006-01-02 15:04:05"),
			"completed_at":   record.CompletedAt.Format("2006-01-02 15:04:05"),
		})
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"data":    result,
	})
}

// 获取单个执行记录详情
func getPipelineExecutionRecord(c *gin.Context) {
	id := c.Param("id")
	user, ok := getCurrentUser(c)
	if !ok {
		c.JSON(http.StatusUnauthorized, gin.H{
			"success": false,
			"message": "用户未认证",
		})
		return
	}

	var record PipelineExecutionRecord
	if err := db.Where("id = ? AND username = ?", id, user.Username).First(&record).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			c.JSON(http.StatusNotFound, gin.H{
				"success": false,
				"message": "执行记录不存在",
			})
		} else {
			logger.Errorf("查询执行记录详情失败: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{
				"success": false,
				"message": "获取执行记录详情失败: " + err.Error(),
			})
		}
		return
	}

	// 解析存储的JSON数据
	var requestData PipelineRequest
	var stepResults interface{}
	var finalOutput interface{}

	json.Unmarshal([]byte(record.RequestData), &requestData)
	json.Unmarshal([]byte(record.StepResults), &stepResults)
	json.Unmarshal([]byte(record.FinalOutput), &finalOutput)

	responseData := map[string]interface{}{
		"id":             record.ID,
		"pipeline_id":    record.PipelineID,
		"pipeline_name":  record.PipelineName,
		"username":       record.Username,
		"status":         record.Status,
		"request_data":   requestData,
		"step_results":   stepResults,
		"final_output":   finalOutput,
		"error_message":  record.ErrorMessage,
		"execution_time": record.ExecutionTime,
		"created_at":     record.CreatedAt,
		"completed_at":   record.CompletedAt,
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"data":    responseData,
	})
}

// 处理单个步骤
func processStep(step StepConfig, index int, mainTable TableData) StepResult {
	// 模拟数据处理逻辑
	sampleData := make([]map[string]interface{}, 0)
	if len(mainTable.PreviewData) > 0 {
		// 取前3行作为样例数据
		for i := 0; i < 3 && i < len(mainTable.PreviewData); i++ {
			sampleData = append(sampleData, mainTable.PreviewData[i])
		}
	}

	// 根据步骤类型模拟不同的输出列
	outputColumns := mainTable.Columns
	switch step.Type {
	case "rename":
		if mapping, ok := step.Config["renameMapping"].(map[string]interface{}); ok {
			newColumns := make([]string, len(outputColumns))
			for i, col := range outputColumns {
				if newName, exists := mapping[col]; exists {
					newColumns[i] = newName.(string)
				} else {
					newColumns[i] = col
				}
			}
			outputColumns = newColumns
		}
	case "feature_derive":
		if newFeatures, ok := step.Config["newFeatures"].([]interface{}); ok {
			for _, feature := range newFeatures {
				outputColumns = append(outputColumns, feature.(string))
			}
		}
	}

	return StepResult{
		StepID:        step.ID,
		Success:       true,
		Message:       fmt.Sprintf("Step %d (%s) executed successfully", index+1, step.Type),
		OutputColumns: outputColumns,
		SampleData:    sampleData,
		DataLength:    len(mainTable.FullData),
		ExecutionTime: "0.5s",
	}
}

func saveTablepiplineConfig(c *gin.Context) {
	log.Printf("begin saveTablepiplineConfig")
	var req SaveTablePiplineConfigRequest
	if err := c.BindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"success": false,
			"message": "Invalid request data",
		})
		return
	}
	log.Printf("begin req %s", req)
	user, ok := getCurrentUser(c)
	if !ok {
		c.JSON(http.StatusUnauthorized, gin.H{
			"success": false,
			"message": "用户未认证",
		})
		return
	}

	// 生成ID如果不存在
	if req.ID == "" {
		req.ID = fmt.Sprintf("config_%d", time.Now().Unix())
	}

	// 序列化配置数据
	stepsJSON, _ := json.Marshal(req.PipelineSteps)
	tablesJSON, _ := json.Marshal(req.Tables)
	outputJSON, _ := json.Marshal(req.OutputConfig)

	// 保存到数据库
	config := TablePipelineConfig{
		ID:            req.ID,
		Name:          req.Name,
		PipelineSteps: string(stepsJSON),
		Tables:        string(tablesJSON),
		OutputConfig:  string(outputJSON),
		Username:      user.Username,
		CreatedAt:     time.Now(),
		UpdatedAt:     time.Now(),
	}

	// 检查是否已存在
	var existingConfig TablePipelineConfig
	if err := db.Where("id = ?", req.ID).First(&existingConfig).Error; err == nil {
		// 更新现有配置
		if err := db.Model(&existingConfig).Updates(map[string]interface{}{
			"name":           req.Name,
			"pipeline_steps": string(stepsJSON),
			"tables":         string(tablesJSON),
			"output_config":  string(outputJSON),
			"updated_at":     time.Now(),
		}).Error; err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"success": false,
				"message": "保存配置失败: " + err.Error(),
			})
			return
		}
	} else {
		// 创建新配置
		if err := db.Create(&config).Error; err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"success": false,
				"message": "保存配置失败: " + err.Error(),
			})
			return
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"message": "Configuration saved successfully",
		"id":      req.ID,
	})
}

// 获取所有配置
func getTablepipelineConfigList(c *gin.Context) {
	log.Printf("begin getConfigList")
	user, ok := getCurrentUser(c)
	if !ok {
		c.JSON(http.StatusUnauthorized, gin.H{
			"success": false,
			"message": "用户未认证",
		})
		return
	}

	var configs []TablePipelineConfig
	if err := db.Where("username = ?", user.Username).Order("updated_at DESC").Find(&configs).Error; err != nil {
		logger.Errorf("查询配置列表失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"success": false,
			"message": "获取配置列表失败: " + err.Error(),
		})
		return
	}

	// 转换为前端需要的格式
	result := make([]map[string]interface{}, 0)
	for _, config := range configs {
		// 解析步骤数量等信息
		var steps []StepConfig
		var tables []TableData
		stepsCount := 0
		tablesCount := 0

		if err := json.Unmarshal([]byte(config.PipelineSteps), &steps); err == nil {
			stepsCount = len(steps)
		}
		if err := json.Unmarshal([]byte(config.Tables), &tables); err == nil {
			tablesCount = len(tables)
		}

		result = append(result, map[string]interface{}{
			"id":   config.ID,
			"name": config.Name,
			// 			"description": config.Description,
			"steps":   stepsCount,
			"tables":  tablesCount,
			"created": config.CreatedAt.Format("2006-01-02 15:04:05"),
			"updated": config.UpdatedAt.Format("2006-01-02 15:04:05"),
		})
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"data":    result,
	})
}

// 添加获取单个配置详情的API
func getTablepiplineConfigDetail(c *gin.Context) {
	id := c.Param("id")
	user, ok := getCurrentUser(c)
	if !ok {
		c.JSON(http.StatusUnauthorized, gin.H{
			"success": false,
			"message": "用户未认证",
		})
		return
	}

	var config TablePipelineConfig
	if err := db.Where("id = ? AND username = ?", id, user.Username).First(&config).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			c.JSON(http.StatusNotFound, gin.H{
				"success": false,
				"message": "配置不存在",
			})
		} else {
			logger.Errorf("查询配置详情失败: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{
				"success": false,
				"message": "获取配置失败: " + err.Error(),
			})
		}
		return
	}

	// 解析配置数据
	var pipelineSteps []StepConfig
	var tables []TableData
	var outputConfig OutputConfig

	if err := json.Unmarshal([]byte(config.PipelineSteps), &pipelineSteps); err != nil {
		logger.Warnf("解析步骤配置失败: %v", err)
		pipelineSteps = []StepConfig{}
	}
	if err := json.Unmarshal([]byte(config.Tables), &tables); err != nil {
		logger.Warnf("解析表数据失败: %v", err)
		tables = []TableData{}
	}
	if err := json.Unmarshal([]byte(config.OutputConfig), &outputConfig); err != nil {
		logger.Warnf("解析输出配置失败: %v", err)
		outputConfig = OutputConfig{Format: "csv", IncludeSteps: false}
	}

	responseData := map[string]interface{}{
		"id":   config.ID,
		"name": config.Name,
		// 		"description":   config.Description,
		"pipelineSteps": pipelineSteps,
		"tables":        tables,
		"outputConfig":  outputConfig,
		"createdAt":     config.CreatedAt,
		"updatedAt":     config.UpdatedAt,
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"data":    responseData,
	})
}

// 删除配置
func deleteTablePiplineConfig(c *gin.Context) {
	id := c.Param("id")
	if _, exists := savedConfigs[id]; !exists {
		c.JSON(http.StatusNotFound, gin.H{
			"success": false,
			"message": "Configuration not found",
		})
		return
	}

	delete(savedConfigs, id)

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"message": "Configuration deleted successfully",
	})
}

type URLLoadRequest struct {
	URL          string `json:"url" binding:"required,url"`
	FileType     string `json:"file_type" binding:"required"`
	PreviewRows  int    `json:"preview_rows"`
	NeedMetadata bool   `json:"need_metadata"`
}

// URL 加载响应结构
type URLLoadResponse struct {
	Tables    []TableInfo `json:"tables"`
	TotalRows int64       `json:"total_rows"`
	Source    string      `json:"source"`
}

// 表格信息结构 - 简化版本
type TableInfo struct {
	Name        string                   `json:"name"`
	FileName    string                   `json:"file_name"`
	SheetName   string                   `json:"sheet_name,omitempty"`
	Columns     []string                 `json:"columns"`
	PreviewData []map[string]interface{} `json:"preview_data"` // 只需要预览数据
	TotalRows   int64                    `json:"total_rows"`   // 总行数信息
	FileSize    int64                    `json:"file_size"`    // 文件大小
	LoadedRows  int                      `json:"loaded_rows"`  // 实际加载的行数（预览行数）
	Filehash    string                   `json:"filehash,omitempty"`
	FilePath    string                   `json:"file_path,omitempty"`
	SourceType  string                   `json:"source_type"` // "url" 或 "file"
}

func parseCSVFileForPreview(filePath string, previewRows int) (TableInfo, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return TableInfo{}, err
	}
	defer file.Close()

	reader := csv.NewReader(file)

	// 读取表头
	headers, err := reader.Read()
	if err != nil {
		return TableInfo{}, err
	}

	// 只读取预览数据
	var previewData []map[string]interface{}
	var previewRowCount int

	for i := 0; i < previewRows; i++ {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			continue // 跳过错误行
		}

		rowData := make(map[string]interface{})
		for j, header := range headers {
			if j < len(record) {
				rowData[header] = record[j]
			} else {
				rowData[header] = ""
			}
		}
		previewData = append(previewData, rowData)
		previewRowCount++
	}

	// 统计总行数（不加载全部数据）
	totalRows, err := countCSVRows(filePath)
	if err != nil {
		// 如果统计失败，使用预览行数作为估计
		totalRows = int64(previewRowCount)
	}

	// 获取文件信息
	fileInfo, err := os.Stat(filePath)
	if err != nil {
		return TableInfo{}, err
	}

	return TableInfo{
		Name:        filepath.Base(filePath),
		FileName:    filepath.Base(filePath),
		Columns:     headers,
		PreviewData: previewData,
		TotalRows:   totalRows,
		FileSize:    fileInfo.Size(),
		LoadedRows:  previewRowCount,
		SourceType:  "url",
	}, nil
}

// 解析 Excel 文件 - 只读取预览数据
func parseExcelFileForPreview(filePath string, previewRows int) ([]TableInfo, error) {
	f, err := excelize.OpenFile(filePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var tables []TableInfo
	sheets := f.GetSheetList()

	for _, sheet := range sheets {
		// 获取所有行
		rows, err := f.GetRows(sheet)
		if err != nil {
			continue
		}

		if len(rows) == 0 {
			continue
		}

		// 表头
		headers := rows[0]

		// 预览数据（最多 previewRows 行）
		var previewData []map[string]interface{}
		loadedRows := 0

		for i := 1; i < len(rows) && i <= previewRows+1; i++ {
			if i >= len(rows) {
				break
			}

			rowData := make(map[string]interface{})
			for j, header := range headers {
				if j < len(rows[i]) {
					rowData[header] = rows[i][j]
				} else {
					rowData[header] = ""
				}
			}
			previewData = append(previewData, rowData)
			loadedRows++
		}

		// 获取文件信息
		fileInfo, err := os.Stat(filePath)
		if err != nil {
			return nil, err
		}

		table := TableInfo{
			Name:        fmt.Sprintf("%s_%s", filepath.Base(filePath), sheet),
			FileName:    filepath.Base(filePath),
			SheetName:   sheet,
			Columns:     headers,
			PreviewData: previewData,
			TotalRows:   int64(len(rows) - 1), // 减去表头
			FileSize:    fileInfo.Size(),
			LoadedRows:  loadedRows,
			SourceType:  "url",
		}

		tables = append(tables, table)
	}

	return tables, nil
}

// 解析 JSON 文件 - 只读取预览数据
func parseJSONFileForPreview(filePath string, previewRows int) (TableInfo, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return TableInfo{}, err
	}
	defer file.Close()

	// 读取文件开头部分来检测JSON结构
	buffer := make([]byte, 1024)
	n, err := file.Read(buffer)
	if err != nil && err != io.EOF {
		return TableInfo{}, err
	}

	// 检测是否是数组JSON
	var jsonData []map[string]interface{}
	var singleObject map[string]interface{}

	if err := json.Unmarshal(buffer[:n], &jsonData); err == nil {
		// 是数组格式 - 只取预览数据
		previewData := make([]map[string]interface{}, 0)
		for i := 0; i < len(jsonData) && i < previewRows; i++ {
			previewData = append(previewData, jsonData[i])
		}

		// 统计总行数
		totalRows, err := countJSONArrayRows(filePath)
		if err != nil {
			totalRows = int64(len(previewData))
		}

		// 获取列名
		var columns []string
		if len(jsonData) > 0 {
			for key := range jsonData[0] {
				columns = append(columns, key)
			}
		}

		fileInfo, _ := os.Stat(filePath)

		return TableInfo{
			Name:        filepath.Base(filePath),
			FileName:    filepath.Base(filePath),
			Columns:     columns,
			PreviewData: previewData,
			TotalRows:   totalRows,
			FileSize:    fileInfo.Size(),
			LoadedRows:  len(previewData),
			SourceType:  "url",
		}, nil
	} else if err := json.Unmarshal(buffer[:n], &singleObject); err == nil {
		// 是单对象格式
		previewData := []map[string]interface{}{singleObject}
		var columns []string
		for key := range singleObject {
			columns = append(columns, key)
		}

		fileInfo, _ := os.Stat(filePath)

		return TableInfo{
			Name:        filepath.Base(filePath),
			FileName:    filepath.Base(filePath),
			Columns:     columns,
			PreviewData: previewData,
			TotalRows:   1,
			FileSize:    fileInfo.Size(),
			LoadedRows:  1,
			SourceType:  "url",
		}, nil
	}

	return TableInfo{}, fmt.Errorf("无法解析JSON文件格式")
}

// 解析表格文件 - 只读取预览数据
func parseTableFileForPreview(filePath, fileType string, previewRows int) ([]TableInfo, error) {
	var tables []TableInfo

	actualFileType := determineFileType(filePath, "")
	if fileType == "auto" {
		fileType = actualFileType
	}

	switch fileType {
	case "csv":
		table, err := parseCSVFileForPreview(filePath, previewRows)
		if err != nil {
			return nil, err
		}
		tables = append(tables, table)
	case "excel":
		excelTables, err := parseExcelFileForPreview(filePath, previewRows)
		if err != nil {
			return nil, err
		}
		tables = append(tables, excelTables...)
	case "json":
		table, err := parseJSONFileForPreview(filePath, previewRows)
		if err != nil {
			return nil, err
		}
		tables = append(tables, table)
	default:
		return nil, fmt.Errorf("不支持的文件类型: %s", fileType)
	}

	return tables, nil
}

func loadTableFromURL(c *gin.Context) {
	logger.Info("=== 开始处理URL数据加载 ===")

	user, ok := getCurrentUser(c)
	if !ok {
		logger.Error("获取当前用户失败")
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "内部服务器错误"})
		return
	}

	var req URLLoadRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		logger.Errorf("请求参数解析失败: %v", err)
		c.JSON(http.StatusBadRequest, gin.H{"status": "error", "message": "请求参数错误: " + err.Error()})
		return
	}

	// 验证 URL
	if !isValidDataURL(req.URL) {
		logger.Errorf("不支持的URL格式: %s", req.URL)
		c.JSON(http.StatusBadRequest, gin.H{"status": "error", "message": "不支持的URL格式"})
		return
	}

	// 创建临时目录
	tempDir := filepath.Join("./projects/data/temp", fmt.Sprintf("url_%d", time.Now().Unix()))
	if err := ensureDir(tempDir); err != nil {
		logger.Errorf("创建临时目录失败: %s, 错误: %v", tempDir, err)
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "创建临时目录失败"})
		return
	}
	defer os.RemoveAll(tempDir) // 清理临时文件

	// 下载文件
	filePath, err := downloadFromURL(req.URL, tempDir)
	if err != nil {
		logger.Errorf("下载文件失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "下载文件失败: " + err.Error()})
		return
	}

	// 计算文件哈希
	fileHash, fileSize, err := calculateFileHashAndSize(filePath)
	if err != nil {
		logger.Errorf("计算文件哈希失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "文件处理失败"})
		return
	}

	// 检查是否已存在该文件
	var existingFile PFileMetadata
	if err := db.Where("md5 = ?", fileHash).First(&existingFile).Error; err == nil {
		logger.Infof("文件已存在: %s", existingFile.Filepath)
		// 使用已存在的文件路径
		filePath = existingFile.Filepath
	} else {
		// 保存到永久存储
		permanentPath := filepath.Join("./projects/data/files", fileHash+filepath.Ext(filePath))
		if err := os.Rename(filePath, permanentPath); err != nil {
			// 如果移动失败，尝试复制
			if err := copyFile(filePath, permanentPath); err != nil {
				logger.Errorf("保存文件失败: %v", err)
				c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "保存文件失败"})
				return
			}
		}
		filePath = permanentPath

		// 保存文件元数据
		fileMeta := PFileMetadata{
			Filename:    filepath.Base(req.URL),
			Filesize:    fileSize,
			MD5:         fileHash,
			UploadTime:  time.Now(),
			ContentType: "application/octet-stream",
			FileType:    determineFileType(req.URL, ""),
			Filepath:    filePath,
			Username:    user.Username,
			SourceURL:   req.URL,
			ExtraInfo:   "{}",
		}

		if err := db.Create(&fileMeta).Error; err != nil {
			logger.Errorf("保存文件元数据失败: %v", err)
		}
	}

	// 解析文件获取预览数据（不加载完整数据）
	tables, err := parseTableFileForPreview(filePath, req.FileType, req.PreviewRows)
	if err != nil {
		logger.Errorf("解析表格文件失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "解析文件失败: " + err.Error()})
		return
	}

	// 构建响应
	response := URLLoadResponse{
		Tables:    tables,
		Source:    req.URL,
		TotalRows: 0,
	}

	for _, table := range tables {
		response.TotalRows += table.TotalRows
	}

	logger.Infof("URL数据加载完成: 加载了 %d 个表，预览 %d 行数据", len(tables), req.PreviewRows)
	c.JSON(http.StatusOK, gin.H{
		"status": "success",
		"data":   response,
	})
}

func downloadFromURL(url, destDir string) (string, error) {
	// 创建 HTTP 客户端
	client := &http.Client{
		Timeout: 30 * time.Minute, // 30分钟超时，支持大文件
	}

	// 创建请求
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return "", fmt.Errorf("创建请求失败: %v", err)
	}

	// 添加一些常见的 headers
	req.Header.Set("User-Agent", "TableDataPipeline/1.0")

	// 执行请求
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("请求失败: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTP错误: %s", resp.Status)
	}

	// 生成文件名
	fileName := generateFileNameFromURL(url)
	filePath := filepath.Join(destDir, fileName)

	// 创建文件
	file, err := os.Create(filePath)
	if err != nil {
		return "", fmt.Errorf("创建文件失败: %v", err)
	}
	defer file.Close()

	// 下载文件
	_, err = io.Copy(file, resp.Body)
	if err != nil {
		return "", fmt.Errorf("下载失败: %v", err)
	}

	return filePath, nil
}
func isValidDataURL(urlStr string) bool {
	parsed, err := url.Parse(urlStr) // 使用 url.Parse 而不是 urlStr.Parse
	if err != nil {
		return false
	}

	// 支持 HTTP/HTTPS
	if parsed.Scheme == "http" || parsed.Scheme == "https" {
		return true
	}

	// 可以扩展支持 S3 等其他协议
	return false
}

func generateFileNameFromURL(urlStr string) string {
	parsed, err := url.Parse(urlStr) // 同样修复这里
	if err != nil {
		return fmt.Sprintf("file_%d", time.Now().Unix())
	}

	name := filepath.Base(parsed.Path)
	if name == "" || name == "." {
		name = fmt.Sprintf("download_%d", time.Now().Unix())
	}

	return name
}

func calculateFileHashAndSize(filePath string) (string, int64, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", 0, err
	}
	defer file.Close()

	hash := md5.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", 0, err
	}

	fileInfo, err := file.Stat()
	if err != nil {
		return "", 0, err
	}

	// 使用 hex.EncodeToString
	return hex.EncodeToString(hash.Sum(nil)), fileInfo.Size(), nil
}

func copyFile(src, dst string) error {
	source, err := os.Open(src)
	if err != nil {
		return err
	}
	defer source.Close()

	destination, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer destination.Close()

	_, err = io.Copy(destination, source)
	return err
}

func countCSVRows(filePath string) (int64, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return 0, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	var rowCount int64 = -1 // 减去表头

	for {
		_, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			continue // 跳过错误行
		}
		rowCount++
	}

	return rowCount, nil
}

func countJSONArrayRows(filePath string) (int64, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return 0, err
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	var count int64

	// 读取开头的 [
	t, err := decoder.Token()
	if err != nil || t != json.Delim('[') {
		return 0, fmt.Errorf("不是有效的JSON数组")
	}

	// 统计对象数量
	for decoder.More() {
		var obj map[string]interface{}
		if err := decoder.Decode(&obj); err != nil {
			break
		}
		count++
	}

	return count, nil
}
