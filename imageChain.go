package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"sort"
	"time"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm"
)

// 处理链表（关联图片hash）
type ImageChain struct {
	ID               uint           `json:"id" gorm:"primaryKey;autoIncrement"`
	Chainname        string         `gorm:"size:100;not null" json:"chainname"` // 处理链名称
	Imagehash        string         `gorm:"size:64;not null" json:"imagehash"`  // 关联的图片hash
	Steps            string         `gorm:"type:text" json:"steps"`             // 处理步骤(JSON字符串)
	Annotations      string         `gorm:"type:text" json:"annotations"`       // 标注信息(JSON字符串)
	Processingresult string         `gorm:"type:text" json:"processingresult"`  // 处理结果(JSON字符串)
	Username         string         `not null" json:"username"`                 // 所属用户
	Createdat        time.Time      `json:"createdat"`
	Updatedat        time.Time      `json:"updatedat"`
	Deletedat        gorm.DeletedAt `gorm:"index" json:"-"`
}

type ImageChainOnceProcessRecord struct {
	ID           uint      `json:"id" gorm:"primaryKey;autoIncrement"`
	Chainname    string    `gorm:"size:100;not null" json:"chainname"` // 处理链名称
	Imagehash    string    `gorm:"size:64;not null" json:"image_hash"` // 处理的图片hash
	Username     string    `not null" json:"username"`                 // 所属用户
	Createdat    time.Time `json:"createdat"`
	Status       string    `gorm:"size:20" json:"status"` // 处理状态
	ErrorMessage string    `gorm:"type:text" json:"error_message"`
	Steps        string    `gorm:"type:text" json:"steps"`       // 处理步骤(JSON字符串)
	Stepsresult  string    `gorm:"type:text" json:"stepsresult"` // 处理结果(JSON字符串)
}

// 用户手动处理结果表（支持计数、量测、画框等手动操作）
type ImageManualProcess struct {
	ID          uint           `gorm:"primaryKey" json:"id"`
	Chainname   string         `gorm:"size:100;not null" json:"chainname"`  // 处理链名称
	Imagehash   string         `gorm:"size:64;not null" json:"imagehash"`   // 关联的图片hash
	Username    string         `gorm:"not null" json:"username"`            // 操作用户
	ProcessType string         `gorm:"size:50;not null" json:"processtype"` // 处理类型：count/measure/box/other
	Content     string         `gorm:"type:text;not null" json:"content"`   // 处理内容（JSON）
	Description string         `gorm:"size:500" json:"description"`         // 处理描述
	Createdat   time.Time      `json:"createdat"`
	Updatedat   time.Time      `json:"updatedat"`
	Deletedat   gorm.DeletedAt `gorm:"index" json:"-"`
}

// 初始化所有表
func InitImageChainTables(db *gorm.DB) error {
	if err := db.AutoMigrate(
		&PFileMetadata{},
		&ImageChain{},
		&ImageChainOnceProcessRecord{},
		&ImageManualProcess{},
	); err != nil {
		return err
	}
	return nil
}

// 注册所有路由
func RegisterAllRoutes(r *gin.Engine) {

	// 处理链相关
	r.POST("/api/chain/save", authMiddleware, saveChain)           // 保存处理链
	r.PUT("/api/chain/update/:id", authMiddleware, updateChain)    // 更新处理链
	r.DELETE("/api/chain/delete/:id", authMiddleware, deleteChain) // 删除处理链
	r.GET("/api/chain/list", authMiddleware, listChains)           // 获取处理链列表

	// 处理执行相关
	r.POST("/api/process-image", authMiddleware, executeChain) // 执行处理链
}

// 简化版本 - 只获取Chainname列表
func listChains(c *gin.Context) {
	logger.Info("=== listChains ===")

	user, ok := getCurrentUser(c)
	if !ok {
		logger.Error("获取当前用户失败")
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "内部服务器错误"})
		return
	}
	logger.Infof("Received request from user: %s", user.Username) // 增加用户日志，便于排查

	// 直接查询该用户所有未删除的链，按创建时间倒序（确保最新的在前面）
	var allChains []ImageChain
	if db != nil {
		logger.Info("Querying chains from database")
		err := db.Where("username = ? AND deletedat IS NULL", user.Username).
			Order("createdat DESC"). // 先按创建时间倒序，确保最新的记录在前面
			Find(&allChains).Error

		if err != nil {
			logger.Errorf("查询处理链失败: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "查询处理链失败"})
			return
		}
	}

	// 按 chainname + username 去重（参考projects的去重逻辑，确保唯一性）
	uniqueChains := make(map[string]ImageChain)
	for _, chain := range allChains {
		// 用 chainname + username 作为唯一键，避免同chainname不同用户的冲突（即使查询条件已过滤用户，仍保持逻辑一致性）
		key := chain.Chainname + "_" + chain.Username
		if _, exists := uniqueChains[key]; !exists {
			uniqueChains[key] = chain // 因前面已按createdat倒序，第一个出现的就是最新记录
		}
	}

	// 转换为结果列表
	var chains []map[string]interface{}
	for _, chain := range uniqueChains {
		chainInfo := map[string]interface{}{
			"chainname": chain.Chainname,
			"steps":     chain.Steps,
			"createdat": chain.Createdat,
			"updatedat": chain.Updatedat,
		}

		// 解析步骤数量
		if chain.Steps != "" {
			var steps []interface{}
			if err := json.Unmarshal([]byte(chain.Steps), &steps); err == nil {
				chainInfo["steps_count"] = len(steps)
			} else {
				chainInfo["steps_count"] = 0
				logger.Warnf("解析步骤失败 for chain %s: %v", chain.Chainname, err) // 增加解析错误日志
			}
		} else {
			chainInfo["steps_count"] = 0
		}

		chains = append(chains, chainInfo)
	}

	// 按创建时间倒序排序（与projects逻辑一致）
	sort.Slice(chains, func(i, j int) bool {
		return chains[i]["createdat"].(time.Time).After(chains[j]["createdat"].(time.Time))
	})

	logger.Infof("Successfully queried %d unique chains", len(chains))
	c.JSON(http.StatusOK, gin.H{
		"status": "success",
		"data": gin.H{
			"chains": chains,
			"total":  len(chains),
		},
	})
}

func ensureDir(path string) error {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return os.MkdirAll(path, 0755) // 权限：所有者可读写执行，其他可读写
	}
	return nil
}

// 辅助函数：返回JSON响应
func jsonResponse(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

// 1. 保存处理链
func saveChain(c *gin.Context) {
	user, ok := getCurrentUser(c)
	if !ok {
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "获取用户信息失败"})
		return
	}

	var req struct {
		ChainName        string        `json:"name" binding:"required"`
		ImageHash        string        `json:"image_hash"` // 改为可选，因为可能没有图片
		Steps            []interface{} `json:"steps" binding:"required"`
		Annotations      interface{}   `json:"annotations"`
		ProcessingResult interface{}   `json:"processing_results"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"status": "error", "message": "参数错误: " + err.Error()})
		return
	}

	// 检查是否已存在同名处理链
	var existingChain ImageChain
	result := db.Where("chainname = ? AND username = ?", req.ChainName, user.Username).First(&existingChain)

	if result.Error == nil {
		// 找到同名处理链，返回冲突错误
		c.JSON(http.StatusConflict, gin.H{
			"status":  "error",
			"message": fmt.Sprintf("处理链名称 '%s' 已存在", req.ChainName),
			"data":    existingChain.ID, // 返回已存在链的ID，便于前端更新
		})
		return
	} else if !errors.Is(result.Error, gorm.ErrRecordNotFound) {
		// 数据库查询错误
		logger.Error("查询处理链失败: ", result.Error)
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "查询失败"})
		return
	}

	// 序列化步骤和标注信息为JSON字符串
	stepsJSON, err := json.Marshal(req.Steps)
	if err != nil {
		logger.Error("序列化步骤失败: ", err)
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "处理步骤格式错误"})
		return
	}

	annotationsJSON, _ := json.Marshal(req.Annotations)
	resultsJSON, _ := json.Marshal(req.ProcessingResult)

	chain := ImageChain{
		Chainname:        req.ChainName,
		Imagehash:        req.ImageHash,
		Steps:            string(stepsJSON),
		Annotations:      string(annotationsJSON),
		Processingresult: string(resultsJSON),
		Username:         user.Username,
		Createdat:        time.Now(),
		Updatedat:        time.Now(),
	}

	if err := db.Create(&chain).Error; err != nil {
		logger.Error("保存处理链失败: ", err)
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "保存失败: " + err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":  "success",
		"message": "处理链保存成功",
		"data":    chain.ID,
	})
}

// 2. 更新处理链
// 2. 更新处理链
func updateChain(c *gin.Context) {
	user, ok := getCurrentUser(c)
	if !ok {
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "获取用户信息失败"})
		return
	}

	chainID := c.Param("id")
	if chainID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"status": "error", "message": "链ID不能为空"})
		return
	}

	var req struct {
		ChainName        string        `json:"name" binding:"required"`
		ImageHash        string        `json:"image_hash"`
		Steps            []interface{} `json:"steps" binding:"required"`
		Annotations      interface{}   `json:"annotations"`
		ProcessingResult interface{}   `json:"processing_results"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"status": "error", "message": "参数错误: " + err.Error()})
		return
	}

	// 检查处理链是否存在且属于当前用户
	var existingChain ImageChain
	if err := db.Where("id = ? AND username = ?", chainID, user.Username).First(&existingChain).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			c.JSON(http.StatusNotFound, gin.H{"status": "error", "message": "处理链不存在"})
		} else {
			c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "查询失败"})
		}
		return
	}

	// 序列化数据
	stepsJSON, err := json.Marshal(req.Steps)
	if err != nil {
		logger.Error("序列化步骤失败: ", err)
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "处理步骤格式错误"})
		return
	}

	annotationsJSON, _ := json.Marshal(req.Annotations)
	resultsJSON, _ := json.Marshal(req.ProcessingResult)

	// 更新处理链
	updates := map[string]interface{}{
		"chainname":        req.ChainName,
		"imagehash":        req.ImageHash,
		"steps":            string(stepsJSON),
		"annotations":      string(annotationsJSON),
		"processingresult": string(resultsJSON),
		"updatedat":        time.Now(),
	}

	if err := db.Model(&existingChain).Updates(updates).Error; err != nil {
		logger.Error("更新处理链失败: ", err)
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "更新失败: " + err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":  "success",
		"message": "处理链更新成功",
		"data":    existingChain.ID,
	})
}

// 3. 删除处理链
func deleteChain(c *gin.Context) {
	id := c.Param("id")
	if err := db.Delete(&ImageChain{}, "id = ?", id).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "删除失败"})
		return
	}
	c.JSON(http.StatusOK, gin.H{"status": "success", "message": "删除成功"})
}

// 4. 执行处理链（关联图片hash）
// 执行处理链（关联图片hash）
func executeChain(c *gin.Context) {
	logger.Info("=== executeChain ===")

	// 获取当前用户名
	user, ok := getCurrentUser(c)
	if !ok {
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "获取用户信息失败"})
		return
	}
	logger.Info("user: ", user.Username)
	var req struct {
		ChainName   string        `json:"chainName"`
		Image       string        `json:"image" binding:"required"` // 可能是hash或base64
		Steps       []interface{} `json:"steps" binding:"required"`
		Annotations interface{}   `json:"annotations"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		logger.Error("参数绑定错误:", err)
		c.JSON(http.StatusBadRequest, gin.H{"status": "error", "message": "参数错误"})
		return
	}

	// 1. 判断图片输入类型并获取图片hash和路径
	var imageHash, imagePath string
	var err error

	if isBase64Image(req.Image) {
		// 如果是base64图片，保存并计算hash（使用MD5）
		imageHash, imagePath, err = saveBase64ImageWithMD5(req.Image)
		if err != nil {
			logger.Error("图片保存失败:", err)
			c.JSON(http.StatusBadRequest, gin.H{"status": "error", "message": "图片处理失败: " + err.Error()})
			return
		}
	} else {
		// 如果是hash，直接使用
		imageHash = req.Image
		// 查询图片路径
		var imgMeta PFileMetadata
		if err := db.Where("hash = ?", imageHash).First(&imgMeta).Error; err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"status": "error", "message": "图片不存在"})
			return
		}
		imagePath = imgMeta.Filepath
	}

	// 2. 将处理步骤转换为JSON字符串
	stepsJSON, err := json.Marshal(req.Steps)
	if err != nil {
		logger.Error("步骤序列化失败:", err)
		c.JSON(http.StatusBadRequest, gin.H{"status": "error", "message": "处理步骤格式错误"})
		return
	}

	// 3. 创建处理记录（初始状态为processing）
	processRecord := ImageChainOnceProcessRecord{
		Chainname: req.ChainName,
		Imagehash: imageHash,
		Username:  user.Username,
		Createdat: time.Now(),
		Status:    "processing",
		Steps:     string(stepsJSON),
	}

	if err := db.Create(&processRecord).Error; err != nil {
		logger.Error("创建处理记录失败:", err)
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "处理记录创建失败"})
		return
	}

	// 4. 异步执行Python处理链
	err, output := executeImageChainPythonProcess(processRecord.ID, imagePath, string(stepsJSON), req.Annotations)
	if err != nil {
		logger.Error("executeImageChainPythonProcess error detail: ", err, output)

		// 更新处理记录状态为失败
		db.Model(&processRecord).Updates(map[string]interface{}{
			"status":        "failed",
			"error_message": fmt.Sprintf("任务执行失败: %s", err),
			"stepsresult":   output,
		})

		c.JSON(http.StatusInternalServerError, gin.H{
			"status":  "error",
			"message": fmt.Sprintf("任务执行失败: %s", err),
			"data":    output, // 返回完整的结构化错误信息
		})
		return
	}

	// 5. 处理成功，更新记录状态
	db.Model(&processRecord).Updates(map[string]interface{}{
		"status":      "success",
		"stepsresult": output,
	})

	// 6. 解析输出结果返回给前端
	var resultData map[string]interface{}
	if err := json.Unmarshal([]byte(output), &resultData); err != nil {
		// 如果无法解析为JSON，直接返回原始输出
		resultData = map[string]interface{}{
			"output": output,
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"status":  "success",
		"message": "处理链执行成功",
		"data":    resultData,
	})
}

// 更新处理记录状态（保持不变）
func updateProcessRecord(recordID uint, status, stepsResult, message string) {
	updates := map[string]interface{}{
		"status":      status,
		"stepsresult": stepsResult,
		"updated_at":  time.Now(),
	}

	if status == "failed" {
		updates["error_message"] = message
	}

	if err := db.Model(&ImageChainOnceProcessRecord{}).Where("id = ?", recordID).Updates(updates).Error; err != nil {
		logger.Errorf("更新处理记录失败 [Record%d]: %v", recordID, err)
	} else {
		logger.Infof("处理记录更新 [Record%d]: status=%s", recordID, status)
	}
}
