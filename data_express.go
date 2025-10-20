package main

import (
	"bytes"
	"database/sql" // 新增这一行，导入标准库的sql包
	"encoding/csv" // 添加csv包
	"encoding/json"
	"errors"
	"fmt"
	"io" // 添加io包
	"log"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
	"gorm.io/gorm"

	"path/filepath" // 添加filepath包
)

// DataFetchTable 数据集市表结构
type DataFetchTable struct {
	ID             uint      `gorm:"primaryKey" json:"id"`
	TableName      string    `gorm:"size:255;not null;uniqueIndex" json:"tableName"` // 表名
	DisplayName    string    `gorm:"size:255" json:"displayName"`                    // 显示名称
	Description    string    `gorm:"size:1000" json:"description"`                   // 描述
	DataSource     string    `gorm:"size:255;not null" json:"dataSource"`            // 数据源
	SQLDefinition  string    `gorm:"type:text" json:"sqlDefinition"`                 // SQL定义
	SchemaJSON     string    `gorm:"type:text" json:"schemaJSON"`                    // 表结构JSON
	AllowedUsers   string    `gorm:"type:text" json:"allowedUsers"`                  // 允许访问的用户列表，逗号分隔
	IsView         bool      `gorm:"default:false" json:"isView"`                    // 是否是视图
	Status         string    `gorm:"size:50;default:'pending'" json:"status"`        // 状态: pending, active, error
	LastAnalyzedAt int64     `json:"lastAnalyzedAt"`                                 // 最后分析时间
	UpdatedAt      time.Time `gorm:"autoUpdateTime" json:"updatedAt"`                // 自动更新修改时间
}

// DataFetchTableSchema 表结构分析结果
type DataFetchTableSchema struct {
	ColumnName string `json:"columnName"` // 列名
	DataType   string `json:"dataType"`   // 数据类型
	IsNullable bool   `json:"isNullable"` // 是否可为空
	Comment    string `json:"comment"`    // 列注释
}

// // 获取所有数据表
// func getDataFetchTables(c *gin.Context) {
// 	var tables []DataFetchTable
// 	if err := db.Find(&tables).Error; err != nil {
// 		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
// 		return
// 	}
// 	c.JSON(http.StatusOK, tables)
// }

// 创建或更新数据表
func saveDataFetchTable(c *gin.Context) {
	log.Printf("begin saveDataFetchTable")

	// body, err := io.ReadAll(c.Request.Body)
	// if err != nil {
	// 	log.Printf("读取请求体失败: %s", err)
	// } else {
	// 	log.Printf("接收到的请求体: %s", string(body)) // 关键：查看是否为空
	// }

	var table DataFetchTable
	if err := c.ShouldBindJSON(&table); err != nil {
		log.Printf("saveDataFetchTable ShouldBindJSON eror:%s", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	user, ok := getCurrentUser(c)
	if !ok {
		logger.Warn("Failed to assert current user type")
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "Failed to assert current user"})
		return
	}
	logger.Infof("Received request from user: %s", user.Username)

	if user.Username == "" {
		logger.Warn("Username is empty")
		c.JSON(http.StatusBadRequest, gin.H{"status": "error", "message": "用户名为空，不合法"})
		return
	}

	// 优化：精确判断当前用户是否已在AllowedUsers中
	targetUser := "," + user.Username + ","
	allowedUsersWithBounds := "," + table.AllowedUsers + ","

	if !strings.Contains(allowedUsersWithBounds, targetUser) {
		if table.AllowedUsers == "" {
			// 空字符串时直接赋值（无多余逗号）
			table.AllowedUsers = user.Username
		} else {
			// 非空时用逗号分隔追加
			table.AllowedUsers += "," + user.Username
		}
	}

	// 保留原有逻辑：更新时保留状态和分析时间
	if table.ID != 0 {
		var existingTable DataFetchTable
		if err := db.First(&existingTable, table.ID).Error; err == nil {
			table.Status = existingTable.Status
			table.LastAnalyzedAt = existingTable.LastAnalyzedAt
			table.SchemaJSON = existingTable.SchemaJSON
		}
	} else {
		table.Status = "pending"
	}

	if err := db.Save(&table).Error; err != nil {
		log.Printf("save error:%s", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "表保存成功", "id": table.ID})
}

// 删除数据表
func deleteDataFetchTable(c *gin.Context) {
	id := c.Param("table_name")
	result := db.Where("table_name = ?", id).Delete(&DataFetchTable{})
	if result.Error != nil {
		logger.WithError(result.Error).Error("删除表失败")
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "删除表失败",
			"details": result.Error.Error(),
		})
		return
	}

	if result.RowsAffected == 0 {
		logger.Warn("没有表被删除")
		c.JSON(http.StatusNotFound, gin.H{
			"error": fmt.Sprintf("表 '%s' 不存在或已被删除", id),
		})
		return
	}

	logger.Info("表删除成功")
	c.JSON(http.StatusOK, gin.H{
		"message": "表删除成功",
		"deleted": gin.H{
			"table_name":    id,
			"rows_affected": result.RowsAffected,
		},
	})
}

func analyzeTableStructure(c *gin.Context) {
	tableName := c.Param("table_name")
	logger := logrus.WithFields(logrus.Fields{
		"endpoint":   "/data-fetch/tables/analyze",
		"table_name": tableName,
	})

	logger.Info("开始处理表结构分析请求")

	// 验证table_name是否为空
	if tableName == "" {
		logger.Warn("缺少表名参数")
		c.JSON(http.StatusBadRequest, gin.H{"error": "缺少表名参数"})
		return
	}

	// 查询表元数据
	var table DataFetchTable
	if err := db.Where("table_name = ?", tableName).First(&table).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			logger.WithError(err).Warn("表不存在")
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("表 '%s' 不存在", tableName)})
		} else {
			logger.WithError(err).Error("查询表失败")
			c.JSON(http.StatusInternalServerError, gin.H{"error": "查询表失败: " + err.Error()})
		}
		return
	}

	logger.Info("开始分析表结构")

	// 获取实际表结构
	schema, err := getTableSchema(db, tableName)
	if err != nil {
		logger.WithError(err).Error("分析表结构失败")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "分析表结构失败: " + err.Error()})
		return
	}

	if len(schema) == 0 {
		logger.Warn("获取到的表结构为空")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取到的表结构为空，请检查表是否存在或是否有权限访问"})
		return
	}

	schemaJSON, err := json.Marshal(schema)
	if err != nil {
		logger.WithError(err).Error("生成表结构JSON失败")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "生成表结构JSON失败: " + err.Error()})
		return
	}

	// 更新表记录
	table.SchemaJSON = string(schemaJSON)
	table.Status = "active"
	table.LastAnalyzedAt = time.Now().Unix()

	if err := db.Save(&table).Error; err != nil {
		logger.WithError(err).Error("保存表结构失败")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "保存表结构失败: " + err.Error()})
		return
	}

	logger.WithField("columns_count", len(schema)).Info("表结构分析完成")
	c.JSON(http.StatusOK, gin.H{
		"message":      "表结构分析完成",
		"schema":       schema,
		"lastAnalyzed": table.LastAnalyzedAt,
	})
}

// getTableSchema 获取实际表结构
func getTableSchema(db *gorm.DB, tableName string) ([]DataFetchTableSchema, error) {
	var schema []DataFetchTableSchema

	// 先打印当前数据库类型
	dbType := db.Dialector.Name()
	log.Printf("当前数据库类型: %s", dbType)

	// 获取当前数据库名
	var dbName string
	if err := db.Raw("SELECT DATABASE()").Scan(&dbName).Error; err != nil {
		log.Printf("获取当前数据库名失败: %v", err)
		dbName = ""
	}
	log.Printf("当前数据库名: %s, 表名: %s", dbName, tableName)

	switch dbType {
	case "mysql":
		// MySQL专用查询
		query := `
            SELECT 
                COLUMN_NAME as column_name,
                DATA_TYPE as data_type,
                IF(IS_NULLABLE = 'YES', true, false) as is_nullable,
                COLUMN_COMMENT as comment
            FROM 
                INFORMATION_SCHEMA.COLUMNS 
            WHERE 
                TABLE_SCHEMA = ? 
                AND TABLE_NAME = ?
            ORDER BY 
                ORDINAL_POSITION`

		if err := db.Raw(query, dbName, tableName).Scan(&schema).Error; err != nil {
			return nil, fmt.Errorf("MySQL查询失败: %v", err)
		}

	case "postgres":
		// PostgreSQL专用查询
		query := `
            SELECT 
                column_name as column_name,
                data_type as data_type,
                (is_nullable = 'YES') as is_nullable,
                '' as comment
            FROM 
                information_schema.columns
            WHERE 
                table_name = $1
            ORDER BY 
                ordinal_position`

		if err := db.Raw(query, tableName).Scan(&schema).Error; err != nil {
			return nil, fmt.Errorf("PostgreSQL查询失败: %v", err)
		}

	case "sqlite":
		// SQLite专用查询
		type sqliteColumn struct {
			Cid     int
			Name    string
			Type    string
			Notnull int
			Dflt    interface{}
			Pk      int
		}

		var columns []sqliteColumn
		// SQLite 的 PRAGMA 不支持参数绑定，必须直接拼接表名
		query := fmt.Sprintf("PRAGMA table_info(%s)", tableName)
		if err := db.Raw(query).Scan(&columns).Error; err != nil {
			return nil, fmt.Errorf("SQLite查询失败: %v", err)
		}

		for _, col := range columns {
			schema = append(schema, DataFetchTableSchema{
				ColumnName: col.Name,
				DataType:   col.Type,
				IsNullable: col.Notnull == 0, // 0 表示允许为NULL，1 表示NOT NULL
				Comment:    "",               // SQLite 表字段注释需要特殊处理，默认空
			})
		}

	default:
		return nil, fmt.Errorf("不支持的数据库类型: %s", dbType)
	}

	// 打印最终解析结果
	log.Printf("解析到的表结构: %+v", schema)

	if len(schema) == 0 {
		return nil, fmt.Errorf("未获取到任何列信息，请检查表名是否正确")
	}

	return schema, nil
}

// 分析表结构
// func analyzeTableStructure(c *gin.Context) {
// 	id := c.Param("id")

// 	// 首先验证ID是否有效
// 	if id == "" {
// 		c.JSON(http.StatusBadRequest, gin.H{"error": "缺少表ID参数"})
// 		return
// 	}

// 	// 尝试将ID转换为uint（假设ID是数字类型）
// 	tableID, err := strconv.ParseUint(id, 10, 64)
// 	if err != nil {
// 		c.JSON(http.StatusBadRequest, gin.H{"error": "无效的表ID格式"})
// 		return
// 	}

// 	var table DataFetchTable
// 	// 使用正确的查询条件
// 	if err := db.Where("id = ?", tableID).First(&table).Error; err != nil {
// 		if errors.Is(err, gorm.ErrRecordNotFound) {
// 			c.JSON(http.StatusNotFound, gin.H{"error": "表不存在"})
// 		} else {
// 			c.JSON(http.StatusInternalServerError, gin.H{"error": "查询表失败: " + err.Error()})
// 		}
// 		return
// 	}

// 	// 这里应该是实际的表结构分析逻辑
// 	// 模拟分析过程，实际项目中应该解析真实的表结构
// 	schema := []DataFetchTableSchema{
// 		{ColumnName: "id", DataType: "int", IsNullable: false, Comment: "主键ID"},
// 		{ColumnName: "name", DataType: "varchar(255)", IsNullable: true, Comment: "名称"},
// 		{ColumnName: "created_at", DataType: "timestamp", IsNullable: false, Comment: "创建时间"},
// 	}

// 	schemaJSON, err := json.Marshal(schema)
// 	if err != nil {
// 		c.JSON(http.StatusInternalServerError, gin.H{"error": "生成表结构JSON失败: " + err.Error()})
// 		return
// 	}

// 	table.SchemaJSON = string(schemaJSON)
// 	table.Status = "active"
// 	table.LastAnalyzedAt = time.Now().Unix()

// 	if err := db.Save(&table).Error; err != nil {
// 		c.JSON(http.StatusInternalServerError, gin.H{"error": "保存表结构失败: " + err.Error()})
// 		return
// 	}

// 	c.JSON(http.StatusOK, gin.H{
// 		"message":      "表结构分析完成",
// 		"schema":       schema,
// 		"lastAnalyzed": table.LastAnalyzedAt,
// 	})
// }

// 检查用户是否有权限访问表
func checkTableAccess(c *gin.Context) {
	tableName := c.Param("tableName")
	username := c.Query("username")

	var table DataFetchTable
	if err := db.Where("table_name = ?", tableName).First(&table).Error; err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "表不存在"})
		return
	}

	// 检查用户是否在允许列表中
	allowedUsers := strings.Split(table.AllowedUsers, ",")
	hasAccess := false
	for _, user := range allowedUsers {
		if strings.TrimSpace(user) == username {
			hasAccess = true
			break
		}
	}

	if !hasAccess {
		c.JSON(http.StatusForbidden, gin.H{"error": "无权访问此表"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"hasAccess": true, "table": table})
}

// 获取用户有权限的表列表
func getUserTables(c *gin.Context) {
	user, ok := getCurrentUser(c)
	if !ok {
		logger.Warn("Failed to assert current user type")
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "Failed to assert current user"})
		return
	}
	logger.Infof("Received request from user: %s", user.Username)

	if user.Username == "" {
		logger.Warn("Username is empty")
		c.JSON(http.StatusBadRequest, gin.H{"status": "error", "message": "用户名为空，不合法"})
		return
	}

	if strings.Contains(user.Username, "%") {
		logger.Warnf("Username contains invalid character: %s", user.Username)
		c.JSON(http.StatusBadRequest, gin.H{"status": "error", "message": "用户名包含不允许的特殊字符"})
		return
	}

	var tables []DataFetchTable
	if err := db.Where("allowed_users LIKE ?", "%"+user.Username+"%").Find(&tables).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, tables)
}

// 执行SQL查询
func executeQuery(c *gin.Context) {
	var req struct {
		SQL    string                 `json:"sql" binding:"required"`
		Params map[string]interface{} `json:"params,omitempty"` // 允许为nil或空
	}

	// 解析请求体
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "请求格式错误: " + err.Error()})
		return
	}

	// 检查敏感操作
	sqlUpper := strings.ToUpper(req.SQL)
	if strings.Contains(sqlUpper, "DELETE") || strings.Contains(sqlUpper, "UPDATE") ||
		strings.Contains(sqlUpper, "INSERT") || strings.Contains(sqlUpper, "DROP") ||
		strings.Contains(sqlUpper, "ALTER") {
		c.JSON(http.StatusForbidden, gin.H{"error": "只允许SELECT查询"})
		return
	}

	var (
		rows *sql.Rows
		err  error
	)

	// 步骤1：提取SQL中的命名参数（:key格式）
	paramRegex := regexp.MustCompile(`:([a-zA-Z0-9_]+)`)
	matches := paramRegex.FindAllStringSubmatch(req.SQL, -1)
	paramNames := make([]string, 0, len(matches))
	for _, m := range matches {
		paramNames = append(paramNames, m[1]) // 提取参数名（如y_thresold）
	}

	// 步骤2：替换命名参数为位置参数（?）
	sqlWithPositional := paramRegex.ReplaceAllString(req.SQL, "?")

	// 步骤3：处理参数（转换为切片，按参数名顺序）
	var args []interface{}
	if len(paramNames) > 0 {
		// 有命名参数时，按参数名顺序从map中取value
		if req.Params == nil || len(req.Params) == 0 {
			c.JSON(http.StatusBadRequest, gin.H{"error": "SQL包含命名参数，但未提供参数"})
			return
		}
		// 按参数名顺序生成切片（确保与?位置对应）
		for _, name := range paramNames {
			val, ok := req.Params[name]
			if !ok {
				c.JSON(http.StatusBadRequest, gin.H{"error": "参数缺失: " + name})
				return
			}
			args = append(args, val)
		}
	} else {
		// 无命名参数时，不传递任何参数（避免空map导致错误）
		args = nil
	}

	// 步骤4：执行查询（使用替换后的SQL和参数切片）
	gormResult := db.Raw(sqlWithPositional, args...)
	if gormResult.Error != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "查询执行失败: " + gormResult.Error.Error()})
		return
	}
	rows, err = gormResult.Rows()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取结果集失败: " + err.Error()})
		return
	}
	defer rows.Close()

	// 后续处理（获取列、解析结果等，保持不变）
	columns, err := rows.Columns()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取列名失败: " + err.Error()})
		return
	}

	columnTypes, err := rows.ColumnTypes()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取列类型失败: " + err.Error()})
		return
	}

	var results []map[string]interface{}
	for rows.Next() {
		scanners := make([]interface{}, len(columns))
		for i := range scanners {
			switch columnTypes[i].DatabaseTypeName() {
			case "DATE", "DATETIME", "TIMESTAMP":
				var t time.Time
				scanners[i] = &t
			case "BINARY", "VARBINARY", "BLOB":
				var b []byte
				scanners[i] = &b
			case "INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT",
				"FLOAT", "DOUBLE", "DECIMAL":
				var num interface{}
				scanners[i] = &num
			default:
				var s sql.NullString
				scanners[i] = &s
			}
		}

		if err := rows.Scan(scanners...); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "解析结果失败: " + err.Error()})
			return
		}

		row := make(map[string]interface{})
		for i, col := range columns {
			val := scanners[i]
			switch v := val.(type) {
			case *time.Time:
				row[col] = v.Format("2006-01-02 15:04:05")
			case *[]byte:
				row[col] = string(*v)
			case *interface{}:
				row[col] = *v
			case *sql.NullString:
				if v.Valid {
					row[col] = v.String
				} else {
					row[col] = nil
				}
			default:
				row[col] = val
			}
		}
		results = append(results, row)
	}

	if err = rows.Err(); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "结果迭代失败: " + err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"columns": columns,
		"data":    results,
		"count":   len(results),
	})
}

// 增强的文件上传处理
func uploadCSV2DB(c *gin.Context) {
	// 1. 获取上传的文件
	file, err := c.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "获取上传文件失败: " + err.Error()})
		return
	}

	// 2. 检查文件扩展名
	ext := strings.ToLower(filepath.Ext(file.Filename))
	if ext != ".csv" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "只支持CSV文件"})
		return
	}

	// 3. 获取表名参数
	tableName := strings.TrimSpace(c.PostForm("tableName"))
	if tableName == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "表名不能为空"})
		return
	}

	// 4. 打开上传的文件
	src, err := file.Open()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "打开文件失败: " + err.Error()})
		return
	}
	defer src.Close()

	// 5. 创建临时表名(防止冲突)
	tempTable := "temp_" + tableName + "_" + time.Now().Format("20060102150405")

	// 6. 删除已存在的临时表(如果存在)
	if err := db.Exec("DROP TABLE IF EXISTS " + tempTable).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "删除旧表失败: " + err.Error()})
		return
	}

	// 7. 读取CSV文件
	reader := csv.NewReader(src)
	reader.FieldsPerRecord = -1 // 允许可变字段数

	// 读取表头
	headers, err := reader.Read()
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "读取CSV表头失败: " + err.Error()})
		return
	}

	// 8. 创建表SQL - 使用反引号代替双引号
	var createSQL strings.Builder
	createSQL.WriteString("CREATE TABLE " + tempTable + " (")
	for _, header := range headers {
		// 使用反引号包裹字段名，修复SQL语法错误
		createSQL.WriteString("`" + strings.TrimSpace(header) + "` TEXT,")
	}
	createSQLStr := createSQL.String()[:createSQL.Len()-1] + ")" // 去除最后一个逗号

	// 9. 执行创建表
	if err := db.Exec(createSQLStr).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "创建表失败: " + err.Error()})
		return
	}

	// 10. 准备批量插入
	batchSize := 100
	var batchValues []interface{}
	var placeholders []string
	rowCount := 0

	// 11. 读取并插入数据
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			continue // 跳过错误行
		}

		// 填充缺失字段为空字符串
		if len(record) < len(headers) {
			record = append(record, make([]string, len(headers)-len(record))...)
		}

		// 将[]string转换为[]interface{}
		var interfaceValues []interface{}
		for _, v := range record {
			interfaceValues = append(interfaceValues, v)
		}

		// 添加到批量
		batchValues = append(batchValues, interfaceValues...)
		placeholders = append(placeholders, "("+strings.Repeat("?,", len(headers)-1)+"?)")
		rowCount++

		// 达到批量大小执行插入
		if len(placeholders) >= batchSize {
			insertSQL := "INSERT INTO " + tempTable + " VALUES " + strings.Join(placeholders, ",")
			if err := db.Exec(insertSQL, batchValues...).Error; err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": "插入数据失败: " + err.Error()})
				return
			}
			// 重置批量
			batchValues = nil
			placeholders = nil
		}
	}

	// 12. 插入剩余数据
	if len(placeholders) > 0 {
		insertSQL := "INSERT INTO " + tempTable + " VALUES " + strings.Join(placeholders, ",")
		if err := db.Exec(insertSQL, batchValues...).Error; err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "插入数据失败: " + err.Error()})
			return
		}
	}

	// 13. 返回成功响应
	c.JSON(http.StatusOK, gin.H{
		"message":  "文件上传并导入成功",
		"table":    tempTable,
		"columns":  headers,
		"rowCount": rowCount,
	})
}
func updateDataFetchTable(c *gin.Context) {
	// 从URL中获取ID
	id, err := strconv.ParseUint(c.Param("id"), 10, 64)
	if err != nil {
		log.Printf("ParseUint eror:%s", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": "无效的ID"})
		return
	}

	// 1. 读取请求体（如需日志）
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		log.Printf("读取请求体失败: %s", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": "读取请求体失败"})
		return
	}
	// 2. 重置请求体（关键步骤），让后续 ShouldBindJSON 能再次读取
	c.Request.Body = io.NopCloser(bytes.NewBuffer(body))
	log.Printf("接收到的请求体: %s", string(body)) // 打印日志

	// 3. 正常绑定数据（此时请求体已重置，可正常读取）
	var table DataFetchTable
	if err := c.ShouldBindJSON(&table); err != nil {
		log.Printf("ShouldBindJSON 错误: %s", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": "数据解析失败: " + err.Error()})
		return
	}

	// 确保ID匹配
	table.ID = uint(id)

	// 调用保存函数处理更新逻辑
	saveDataFetchTable(c)
}

// 注册路由
func registerDataFetchRoutes(r *gin.RouterGroup) {
	r.GET("/data-fetch/tables", authMiddleware, getUserTables)
	r.GET("/data/tables", authMiddleware, getUserTables)
	r.POST("/data-fetch/tables", authMiddleware, saveDataFetchTable)
	r.POST("/data-fetch/updatetable", authMiddleware, saveDataFetchTable)
	r.DELETE("/data-fetch/tables/:table_name", deleteDataFetchTable)
	r.POST("/data-fetch/tables/analyze/:table_name", analyzeTableStructure)
	r.GET("/data-fetch/tables/access/:tableName", checkTableAccess)

	r.POST("/data/sqltestquery", authMiddleware, executeQuery)
	r.POST("/data/upload2db", authMiddleware, uploadCSV2DB)
}
