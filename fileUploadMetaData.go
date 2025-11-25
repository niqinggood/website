package main

import (
	"crypto/md5"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm"
)

// 图片元信息表（存储图片基本信息和hash）
// 通用文件元数据结构
type PFileMetadata struct {
	ID          uint      `json:"id" gorm:"primaryKey;autoIncrement"`
	Filename    string    `json:"filename"`
	Filesize    int64     `json:"filesize"`
	MD5         string    `json:"md5" gorm:"uniqueIndex"`
	UploadTime  time.Time `json:"upload_time"`
	ContentType string    `json:"content_type,omitempty"`
	FileType    string    `json:"file_type"` // 文件类型: image, csv, excel, json, etc.
	Filepath    string    `json:"filepath"`
	Username    string    `json:"username" gorm:"not null"`
	SourceURL   string    `json:"source_url"`                            // 添加这个字段
	ExtraInfo   string    `json:"extra_info,omitempty" gorm:"type:json"` // 存储额外信息，如图片尺寸、表格列信息等

	// 对于图片类型的额外字段（可选，也可以放在 ExtraInfo 中）
	Width  int `json:"width,omitempty"`
	Height int `json:"height,omitempty"`

	Deletedat gorm.DeletedAt `gorm:"index" json:"-"`
}

// 额外信息结构（存储在 ExtraInfo 字段中）
type FileExtraInfo struct {
	// 图片相关
	Width  int `json:"width,omitempty"`
	Height int `json:"height,omitempty"`

	// 表格相关
	Columns    []string `json:"columns,omitempty"`
	RowCount   int      `json:"row_count,omitempty"`
	SheetNames []string `json:"sheet_names,omitempty"` // Excel 文件

	// 其他通用信息
	Description string `json:"description,omitempty"`
}

// 初始化所有表
func InitfileUploadMetaDataTables(db *gorm.DB) error {
	if err := db.AutoMigrate(
		&PFileMetadata{},
	); err != nil {
		return err
	}
	return nil
}

const imageRootDir = "./projects/data/image"

func RegisterfileMetaRouter(r *gin.Engine) {
	// 图片相关
	r.POST("/api/image/upload", authMiddleware, uploadImage)     // 上传图片并获取hash
	r.GET("/api/image/info/:hash", authMiddleware, getImageInfo) // 获取图片信息
}

func uploadImage(c *gin.Context) {
	logger.Info("=== 开始处理图片上传 ===")
	user, ok := getCurrentUser(c)
	if !ok {
		logger.Error("获取当前用户失败")
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "内部服务器错误"})
		return
	}
	logger.Infof("用户信息: %s", user.Username)

	// 1. 接收上传的文件
	file, err := c.FormFile("file")
	if err != nil {
		logger.Errorf("获取上传文件失败: %v", err)
		c.JSON(http.StatusBadRequest, gin.H{"status": "error", "message": "获取文件失败: " + err.Error()})
		return
	}
	logger.Infof("接收到文件: 文件名=%s, 大小=%d", file.Filename, file.Size)

	// 2. 获取前端传递的元信息
	fileName := c.PostForm("file_name")
	if fileName == "" {
		fileName = file.Filename // 如果前端没传，使用原始文件名
	}
	logger.Infof("使用文件名: %s", fileName)

	// 3. 打开上传的临时文件，计算实际hash
	srcFile, err := file.Open()
	if err != nil {
		logger.Errorf("打开临时文件失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "打开文件失败"})
		return
	}
	defer srcFile.Close()

	// 计算实际hash
	logger.Info("开始计算文件MD5...")
	actualHash, err := calculateFileMD5(srcFile)
	if err != nil {
		logger.Errorf("计算文件MD5失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "计算文件hash失败"})
		return
	}
	logger.Infof("文件MD5计算完成: %s", actualHash)

	// 4. 创建存储目录
	imageDir := filepath.Join(imageRootDir)
	logger.Infof("创建存储目录: %s", imageDir)

	if err := ensureDir(imageDir); err != nil {
		logger.Errorf("创建目录失败: %s, 错误: %v", imageDir, err)
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "创建目录失败: " + err.Error()})
		return
	}
	logger.Info("目录创建成功")

	// 5. 保存文件
	ext := filepath.Ext(fileName)
	dstPath := filepath.Join(imageDir, actualHash+ext)
	logger.Infof("目标文件路径: %s", dstPath)

	// 检查文件是否已存在（避免重复存储）
	if _, err := os.Stat(dstPath); err == nil {
		logger.Infof("文件已存在，检查数据库记录: %s", dstPath)
		var imgMeta PFileMetadata
		if err := db.Where("md5 = ?", actualHash).First(&imgMeta).Error; err == nil {
			logger.Infof("找到已存在的文件记录: ID=%d", imgMeta.ID)
			c.JSON(http.StatusOK, gin.H{
				"status":  "success",
				"message": "文件已存在",
				"data": gin.H{
					"hash":      actualHash,
					"file_name": imgMeta.Filename,
					"id":        imgMeta.ID,
					"filepath":  imgMeta.Filepath,
				},
			})
			return
		} else {
			logger.Warnf("文件存在但数据库记录未找到: %v", err)
		}
	} else {
		logger.Info("文件不存在，准备保存新文件")
	}

	// 保存文件到目标路径
	logger.Info("开始保存文件到目标路径...")
	if err := c.SaveUploadedFile(file, dstPath); err != nil {
		logger.Errorf("保存文件失败: %s, 错误: %v", dstPath, err)
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "保存文件失败: " + err.Error()})
		return
	}
	logger.Info("文件保存成功")

	// 验证文件是否真的保存成功
	if fileInfo, err := os.Stat(dstPath); err != nil {
		logger.Errorf("文件保存后验证失败: %s, 错误: %v", dstPath, err)
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "文件保存验证失败"})
		return
	} else {
		logger.Infof("文件验证成功: 大小=%d bytes", fileInfo.Size())
	}

	// 6. 获取图片尺寸（可选）- 这里可以后续实现
	width, height := 0, 0
	logger.Infof("图片尺寸: %dx%d", width, height)

	// 7. 保存元信息到数据库 - 修复ID问题
	imgMeta := PFileMetadata{
		// ID 由数据库自动生成，不手动设置
		MD5:         actualHash,
		Filename:    fileName,
		Filesize:    file.Size,
		Width:       width,
		Height:      height,
		UploadTime:  time.Now(),
		Username:    user.Username,
		ContentType: file.Header.Get("Content-Type"),
		Filepath:    dstPath,
	}

	logger.Info("开始保存元信息到数据库...")
	logger.Infof("准备插入的数据: MD5=%s, Filename=%s, Filesize=%d",
		imgMeta.MD5, imgMeta.Filename, imgMeta.Filesize)

	// 使用 Create 插入，让 GORM 处理ID生成
	if err := db.Create(&imgMeta).Error; err != nil {
		logger.Errorf("保存元信息到数据库失败: %v", err)
		// 数据库保存失败，删除已上传的文件
		if removeErr := os.Remove(dstPath); removeErr != nil {
			logger.Errorf("删除已上传文件失败: %s, 错误: %v", dstPath, removeErr)
		} else {
			logger.Info("已删除上传的文件")
		}
		c.JSON(http.StatusInternalServerError, gin.H{
			"status":  "error",
			"message": "保存元信息失败: " + err.Error(),
		})
		return
	}
	logger.Infof("元信息保存成功: ID=%s", imgMeta.ID)

	// 8. 返回成功响应
	logger.Info("=== 图片上传处理完成 ===")
	c.JSON(http.StatusOK, gin.H{
		"status":  "success",
		"message": "文件上传成功",
		"data": gin.H{
			"hash":      actualHash,
			"file_name": fileName,
			"id":        imgMeta.ID,
			"path":      dstPath,
			"file_size": file.Size,
			"url":       dstPath,
		},
	})
}

// 上传表格文件
func uploadTableFile(c *gin.Context) {
	logger.Info("=== 开始处理表格文件上传 ===")

	user, ok := getCurrentUser(c)
	if !ok {
		logger.Error("获取当前用户失败")
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "内部服务器错误"})
		return
	}

	// 接收上传的文件
	file, err := c.FormFile("file")
	if err != nil {
		logger.Errorf("获取上传文件失败: %v", err)
		c.JSON(http.StatusBadRequest, gin.H{"status": "error", "message": "获取文件失败: " + err.Error()})
		return
	}

	// 打开文件计算MD5
	srcFile, err := file.Open()
	if err != nil {
		logger.Errorf("打开临时文件失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "打开文件失败"})
		return
	}
	defer srcFile.Close()

	// 计算文件MD5
	fileHash, err := calculateFileMD5(srcFile)
	if err != nil {
		logger.Errorf("计算文件MD5失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "计算文件hash失败"})
		return
	}

	// // 创建存储目录
	fileDir := filepath.Join("./projects/data/files")
	if err := ensureDir(fileDir); err != nil {
		logger.Errorf("创建目录失败: %s, 错误: %v", fileDir, err)
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "创建目录失败: " + err.Error()})
		return
	}

	// 保存文件
	ext := filepath.Ext(file.Filename)
	dstPath := filepath.Join(fileDir, fileHash+ext)

	// 检查是否已存在
	var existingFile PFileMetadata
	if err := db.Where("md5 = ?", fileHash).First(&existingFile).Error; err == nil {
		logger.Infof("文件已存在: %s", existingFile.Filepath)
		c.JSON(http.StatusOK, gin.H{
			"status": "success",
			"data": gin.H{
				"hash":      fileHash,
				"file_name": existingFile.Filename,
				"file_path": existingFile.Filepath,
				"file_type": existingFile.FileType,
			},
		})
		return
	}

	// 保存新文件
	if err := c.SaveUploadedFile(file, dstPath); err != nil {
		logger.Errorf("保存文件失败: %s, 错误: %v", dstPath, err)
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "保存文件失败: " + err.Error()})
		return
	}

	// 确定文件类型
	fileType := determineFileType(file.Filename, file.Header.Get("Content-Type"))

	// 解析表格文件获取基本信息（列信息等）
	extraInfo := FileExtraInfo{}
	if fileType == "csv" || fileType == "excel" {
		// 这里可以调用解析函数获取表格的列信息
		// columns, rowCount := parseTableFile(dstPath, fileType)
		// extraInfo.Columns = columns
		// extraInfo.RowCount = rowCount
	}

	extraInfoJSON, _ := json.Marshal(extraInfo)

	// 保存文件元数据
	fileMeta := PFileMetadata{
		Filename:    file.Filename,
		Filesize:    file.Size,
		MD5:         fileHash,
		UploadTime:  time.Now(),
		ContentType: file.Header.Get("Content-Type"),
		FileType:    fileType,
		Filepath:    dstPath,
		Username:    user.Username,
		ExtraInfo:   string(extraInfoJSON),
	}

	if err := db.Create(&fileMeta).Error; err != nil {
		logger.Errorf("保存文件元数据失败: %v", err)
		// 删除已上传的文件
		os.Remove(dstPath)
		c.JSON(http.StatusInternalServerError, gin.H{
			"status":  "error",
			"message": "保存文件元数据失败: " + err.Error(),
		})
		return
	}

	logger.Info("=== 表格文件上传处理完成 ===")
	c.JSON(http.StatusOK, gin.H{
		"status": "success",
		"data": gin.H{
			"hash":       fileHash,
			"file_name":  file.Filename,
			"file_path":  dstPath,
			"file_type":  fileType,
			"file_size":  file.Size,
			"extra_info": extraInfo,
		},
	})
}

func determineFileType(filename, contentType string) string {
	ext := strings.ToLower(filepath.Ext(filename))

	switch ext {
	case ".csv":
		return "csv"
	case ".xlsx", ".xls":
		return "excel"
	case ".json":
		return "json"
	case ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp":
		return "image"
	case ".txt":
		return "text"
	case ".pdf":
		return "pdf"
	default:
		if strings.Contains(contentType, "spreadsheet") {
			return "excel"
		} else if strings.Contains(contentType, "csv") {
			return "csv"
		} else if strings.Contains(contentType, "json") {
			return "json"
		}
		return "other"
	}
}

func getImageInfo(c *gin.Context) {
	hash := c.Param("hash")
	var imgMeta PFileMetadata
	if err := db.Where("hash = ?", hash).First(&imgMeta).Error; err != nil {
		c.JSON(http.StatusNotFound, gin.H{"status": "error", "message": "图片不存在"})
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"status": "success",
		"data":   imgMeta,
	})
}

func calculateFileMD5(file multipart.File) (string, error) {
	hash := md5.New()

	// 将文件指针重置到开头
	if _, err := file.Seek(0, 0); err != nil {
		return "", fmt.Errorf("failed to seek file: %v", err)
	}

	// 复制文件内容到hash
	if _, err := io.Copy(hash, file); err != nil {
		return "", fmt.Errorf("failed to calculate MD5: %v", err)
	}

	// 再次重置文件指针到开头，以便后续使用
	if _, err := file.Seek(0, 0); err != nil {
		return "", fmt.Errorf("failed to reset file position: %v", err)
	}

	return hex.EncodeToString(hash.Sum(nil)), nil
}

func getImageMetadata(filePath string) (*PFileMetadata, error) {
	fileInfo, err := os.Stat(filePath)
	if err != nil {
		return nil, err
	}

	// 计算MD5
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	fileMD5, err := calculateFileMD5(file)
	if err != nil {
		return nil, err
	}

	return &PFileMetadata{

		Filename:   filepath.Base(filePath), // 修正：使用 Filename
		Filesize:   fileInfo.Size(),         // 修正：使用 Filesize
		MD5:        fileMD5,
		UploadTime: time.Now(),
	}, nil
}

func handleImageUpload(w http.ResponseWriter, r *http.Request) {
	// 解析multipart表单
	err := r.ParseMultipartForm(32 << 20) // 32MB
	if err != nil {
		http.Error(w, "Failed to parse form", http.StatusBadRequest)
		return
	}

	// 获取上传的文件
	file, header, err := r.FormFile("image")
	if err != nil {
		http.Error(w, "Failed to get uploaded file", http.StatusBadRequest)
		return
	}
	defer file.Close()

	// 保存文件
	uploadDir := "./uploads"
	imgMeta, err := saveUploadedImage(file, header, uploadDir)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to save image: %v", err), http.StatusInternalServerError)
		return
	}

	// 返回成功响应
	jsonResponse(w, http.StatusOK, map[string]interface{}{
		"success": true,
		"message": "Image uploaded successfully",
		"data":    imgMeta,
	})
}

func saveUploadedImage(file multipart.File, header *multipart.FileHeader, uploadDir string) (*PFileMetadata, error) {
	// 确保上传目录存在
	if err := os.MkdirAll(uploadDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create upload directory: %v", err)
	}

	// 计算文件MD5
	fileMD5, err := calculateFileMD5(file)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate file MD5: %v", err)
	}

	// 生成唯一文件名
	fileExt := filepath.Ext(header.Filename)
	newFilename := fmt.Sprintf("%s%s", fileMD5, fileExt)
	filePath := filepath.Join(uploadDir, newFilename)

	// 检查文件是否已存在
	if _, err := os.Stat(filePath); err == nil {
		// 文件已存在，直接返回元数据
		fileInfo, _ := os.Stat(filePath)
		return &PFileMetadata{

			Filename:   newFilename,     // 修正：使用 Filename
			Filesize:   fileInfo.Size(), // 修正：使用 Filesize
			MD5:        fileMD5,
			UploadTime: time.Now(),
			Filepath:   filePath,
		}, nil
	}

	// 创建目标文件
	dst, err := os.Create(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to create file: %v", err)
	}
	defer dst.Close()

	// 将上传的文件内容复制到目标文件
	if _, err := io.Copy(dst, file); err != nil {
		return nil, fmt.Errorf("failed to save file: %v", err)
	}

	// 获取文件信息
	fileInfo, err := os.Stat(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to get file info: %v", err)
	}

	// 创建图像元数据
	imgMeta := &PFileMetadata{

		Filename:   newFilename,     // 修正：使用 Filename
		Filesize:   fileInfo.Size(), // 修正：使用 Filesize
		MD5:        fileMD5,
		UploadTime: time.Now(),
		Filepath:   filePath,
	}

	// 这里可以添加获取图像尺寸的代码
	// imgMeta.Width, imgMeta.Height = getImageDimensions(filePath)

	return imgMeta, nil
}

// 保存base64图片并返回MD5 hash和路径（参考你的上传逻辑）
func saveBase64ImageWithMD5(base64Data string) (string, string, error) {
	// 解析base64数据
	parts := strings.Split(base64Data, ",")
	if len(parts) != 2 {
		return "", "", fmt.Errorf("invalid base64 data")
	}

	// 解码base64
	imageData, err := base64.StdEncoding.DecodeString(parts[1])
	if err != nil {
		return "", "", fmt.Errorf("base64解码失败: %v", err)
	}

	// 计算MD5 hash（参考你的calculateFileMD5逻辑）
	hash := md5.Sum(imageData)
	hashStr := hex.EncodeToString(hash[:])

	// 创建保存目录
	uploadDir := "uploads/images"
	if err := os.MkdirAll(uploadDir, 0755); err != nil {
		return "", "", fmt.Errorf("创建目录失败: %v", err)
	}

	// 生成文件路径和扩展名
	fileExt := getFileExtensionFromBase64(parts[0])
	filename := fmt.Sprintf("%s%s", hashStr, fileExt)
	filepath := path.Join(uploadDir, filename)

	// 检查文件是否已存在（参考你的逻辑）
	if _, err := os.Stat(filepath); err == nil {
		// 文件已存在，直接返回
		return hashStr, filepath, nil
	}

	// 保存文件
	if err := os.WriteFile(filepath, imageData, 0644); err != nil {
		return "", "", fmt.Errorf("保存文件失败: %v", err)
	}

	// 创建或更新图片元数据记录
	fileInfo, err := os.Stat(filepath)
	if err != nil {
		return "", "", fmt.Errorf("获取文件信息失败: %v", err)
	}

	// 保存到ImageMetadata表
	imgMeta := PFileMetadata{
		MD5:      hashStr,
		Filepath: filepath,
		Filename: filename,
		Filesize: fileInfo.Size(),
		// 根据你的ImageMetadata结构添加其他字段
		UploadTime: time.Now(),
	}

	// 检查是否已存在记录
	var existingMeta PFileMetadata
	if err := db.Where("hash = ?", hashStr).First(&existingMeta).Error; err != nil {
		// 不存在则创建
		if err := db.Create(&imgMeta).Error; err != nil {
			logger.Warnf("创建图片元数据失败: %v", err)
		}
	}

	return hashStr, filepath, nil
}

// 从base64头部信息获取文件扩展名
func getFileExtensionFromBase64(base64Header string) string {
	switch {
	case strings.Contains(base64Header, "image/png"):
		return ".png"
	case strings.Contains(base64Header, "image/gif"):
		return ".gif"
	case strings.Contains(base64Header, "image/webp"):
		return ".webp"
	case strings.Contains(base64Header, "image/bmp"):
		return ".bmp"
	default:
		return ".jpg" // 默认jpg
	}
}

// 计算MD5 hash（参考你的calculateFileMD5函数）
func calculateMD5(data []byte) string {
	hash := md5.Sum(data)
	return hex.EncodeToString(hash[:])
}

// 判断是否为base64图片
func isBase64Image(data string) bool {
	return strings.Contains(data, "data:image") && strings.Contains(data, "base64,")
}
