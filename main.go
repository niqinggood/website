package main

import (
	"bufio"
	"bytes"
	"embed"
	"encoding/csv"
	"math"
	"regexp"
	"sort"

	"gonum.org/v1/gonum/floats"

	// 	"crypto"
	"crypto/aes"
	"crypto/cipher"

	"path/filepath"
	"reflect"
	"strconv"

	// 	"crypto/rsa"
	"crypto/sha256"
	// 	"crypto/x509"
	"io"
	// 	"io/fs"
	"encoding/base64"
	"io/fs"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"net/url"
	"strings"

	// 	"encoding/csv"

	"encoding/hex"
	"encoding/json"

	// 	"encoding/pem"
	"flag"
	"fmt"
	"os/exec"
	"runtime"

	// 	"net/http"
	"errors"
	"os"
	"time"

	"github.com/gin-gonic/gin"
	"gorm.io/driver/mysql"
	"gorm.io/gorm" // GORM 核心库

	"crypto/rand" // 提供 rand.Reader
	"database/sql/driver"

	"github.com/glebarez/sqlite"
	"github.com/golang-jwt/jwt/v5"

	// 	"github.com/minio/minio-go/v7"
	// 	"github.com/minio/minio-go/v7/pkg/credentials"
	"github.com/sirupsen/logrus"
	// 	"github.com/xuri/excelize/v2"
	"gopkg.in/ini.v1"
	"gorm.io/datatypes"

	// 	"gonum.org/v1/gonum/stat/classif"
	"github.com/robfig/cron/v3"
	"gonum.org/v1/gonum/mat"  // 矩阵操作（仍在主模块）
	"gonum.org/v1/gonum/stat" // 辅助统计函数
	// 	"github.com/sjwhitworth/golearn/base"
	// 	"github.com/sjwhitworth/golearn/ensemble"
	// 	"github.com/sjwhitworth/golearn/trees"
	//      "github.com/sjwhitworth/golearn/base"
	// 	"github.com/go-gota/gota/dataframe"  // 类似 Pandas 的 DataFrame
	//     "github.com/go-gota/gota/series"    // 数据处理
	//     "github.com/montanaflynn/stats"     // 统计分析
	//     "github.com/sajari/regression"      // 回归分析
)

// type DataFactoryFetchDataConfig struct {
// 	Configname   string         `gorm:"primaryKey;column:configname;type:varchar(255);unique" json:"configname"`
// 	Timerange    string         `gorm:"column:timerange;type:varchar(100)" json:"timerange"`
// 	Dates        datatypes.JSON `gorm:"column:dates;type:json" json:"dates"`
// 	Selections   datatypes.JSON `gorm:"column:selections;type:json" json:"selections"`
// 	Detail       datatypes.JSON `gorm:"column:detail;type:json" json:"detail"`
// 	Savedat      string         `gorm:"column:savedat;type:varchar(50)" json:"savedat"`
// 	Layoutconfig datatypes.JSON `gorm:"column:layoutconfig;type:json" json:"layoutconfig"`
// 	Modifier     string         `gorm:"column:modifier;type:varchar(100)" json:"modifier"` // 添加修改者字段
// }

type DataFactoryFetchDataConfig struct {
	Configname   string         `gorm:"primaryKey;column:configname;type:varchar(255);unique" json:"configname"`
	Timerange    string         `gorm:"column:timerange;type:varchar(100)" json:"timerange"`
	Dates        datatypes.JSON `gorm:"column:dates;type:json" json:"dates"`
	Selections   datatypes.JSON `gorm:"column:selections;type:json" json:"selections"`
	Detail       datatypes.JSON `gorm:"column:detail;type:json" json:"detail"`
	SqlJson      datatypes.JSON `gorm:"column:sql_json;type:json" json:"sql_json"` // 新增：存储SQL配置
	Savedat      string         `gorm:"column:savedat;type:varchar(50)" json:"savedat"`
	Layoutconfig datatypes.JSON `gorm:"column:layoutconfig;type:json" json:"layoutconfig"`
	Modifier     string         `gorm:"column:modifier;type:varchar(100)" json:"modifier"`
	Type         string         `gorm:"column:type;type:varchar(50)" json:"type"`               // 新增：区分配置类型（normal/sql）
	LayoutName   string         `gorm:"column:layout_name;type:varchar(50)" json:"layout_name"` //
}

var (
	db     *gorm.DB
	logger = logrus.New()
	// saEngine           *gorm.DB  // 模拟 Flask 中的 sa_engine
	cfg                *ini.File // 全局配置文件对象
	defaultOptionsData map[string]interface{}
	de_engine          *gorm.DB
	run_mode           string

	// 任务ID与定时任务EntryID的映射，用于更新和删除定时任务
	cronScheduler *cron.Cron                      // 重命名变量避免与包名冲突
	taskEntryMap  = make(map[string]cron.EntryID) // 任务ID与定时任务EntryID的映射
	server_ip     string
	server_port   string
)

// encrypt 加密函数
func encrypt(data []byte) ([]byte, error) {
	block, err := aes.NewCipher(encryptionKey)
	if err != nil {
		return nil, err
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}
	nonce := make([]byte, gcm.NonceSize())
	if _, err = io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, err
	}
	ciphertext := gcm.Seal(nonce, nonce, data, nil)
	return ciphertext, nil
}

// decrypt 解密函数
func decrypt(ciphertext []byte) ([]byte, error) {
	block, err := aes.NewCipher(encryptionKey)
	if err != nil {
		return nil, err
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}
	nonceSize := gcm.NonceSize()
	if len(ciphertext) < nonceSize {
		return nil, errors.New("ciphertext too short")
	}
	nonce, ciphertext := ciphertext[:nonceSize], ciphertext[nonceSize:]
	plaintext, err := gcm.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, err
	}
	return plaintext, nil
}

// calculateChecksum 计算校验和
func calculateChecksum(data []byte) string {
	hash := sha256.Sum256(data)
	return fmt.Sprintf("%x", hash)
}

// 加密密钥，长度必须为 16、24 或 32 字节
var encryptionKey = []byte("niqinggoodpatrick&yvonne")

// EncryptedString 自定义类型，用于存储加密字符串
type EncryptedString string

// Value 实现 driver.Valuer 接口，用于在写入数据库时加密数据
func (es EncryptedString) Value() (driver.Value, error) {
	if string(es) == "" {
		return nil, nil
	}
	encrypted, err := encrypt([]byte(es))
	if err != nil {
		return nil, err
	}
	return base64.StdEncoding.EncodeToString(encrypted), nil
}

// Scan 实现 sql.Scanner 接口，用于在从数据库读取数据时解密数据
func (es *EncryptedString) Scan(value interface{}) error {
	log.Printf("Scanning password value: type=%T, value=%v", value, value)

	if value == nil {
		*es = ""
		return nil
	}

	var str string
	switch v := value.(type) {
	case []byte:
		str = string(v)
	case string:
		str = v
	default:
		return fmt.Errorf("unsupported type: %T", value)
	}

	log.Printf("Raw password value from DB: %s", str)

	// 如果是空字符串，直接返回
	if str == "" {
		*es = ""
		return nil
	}

	// 尝试解密
	encrypted, err := base64.StdEncoding.DecodeString(str)
	if err != nil {
		log.Printf("Not base64 encoded, storing raw value")
		*es = EncryptedString(str)
		return nil
	}

	decrypted, err := decrypt(encrypted)
	if err != nil {
		log.Printf("Decryption failed: %v", err)
		return fmt.Errorf("decryption failed: %w", err)
	}

	*es = EncryptedString(decrypted)
	return nil
}

// DataFactorySSOConfig 飞行员单点登录配置结构体
type DataFactorySSOConfig struct {
	SSOUrl        string    `json:"ssoUrl"`
	Method        string    `json:"method"`
	UserField     string    `json:"userField"`
	PasswordField string    `json:"passwordField"`
	SsoSussJudge  string    `json:"ssoSussJudge"`
	UrlEncode     bool      `json:"urlEncode"`
	Checksum      string    `gorm:"column:checksum"`
	Timestamp     time.Time `gorm:"column:timestamp"`
}

type DataFactoryMessage struct {
	ID        uint      `gorm:"primaryKey"`
	Title     string    `json:"title"`
	Content   string    `json:"content"`
	Type      string    `json:"type"`
	Reciver   string    `json:"reciver"`
	Status    string    `json:"status" gorm:"default:'unread'"`
	CreatedAt time.Time `json:"created_at" gorm:"autoCreateTime"`
	CreatedBy string    `json:"created_by"`
}

type DataFactoryMenuItem struct {
	ID         uint   `gorm:"primaryKey"`
	Itemkey    string `json:"itemkey" gorm:"size:50;uniqueIndex;not null"`
	Text       string `json:"text" gorm:"size:100;not null"`
	Icon       string `json:"icon" gorm:"size:50"`
	Link       string `json:"link" gorm:"size:255;not null"`
	IsExternal bool   `json:"isExternal" gorm:"default:false"`
	Useplace   string `json:"useplace" gorm:"size:255;not null"`
	Order      int    `json:"order" gorm:"default:0"`
	Tooltip    string `json:"tooltip" gorm:"size:255"` // 新增字段，用于前端tooltip
	Color      string `json:"color" gorm:"size:50"`    // 新增字段，用于图标颜色
}

var menuItems []DataFactoryMenuItem

func saveMessage(c *gin.Context) {
	var message DataFactoryMessage
	if err := c.ShouldBindJSON(&message); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	logger.Infof("数据库初始化: %s", message)

	// 设置创建时间和创建人
	message.CreatedAt = time.Now()
	message.CreatedBy = "admin" // 这里可以根据实际用户信息设置

	// 保存消息
	if err := db.Create(&message).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// 广播消息（这里可以根据实际需求实现）
	broadcastMessage(message)

	c.JSON(http.StatusOK, gin.H{"status": "success", "message": "消息发送成功"})
}
func broadcastMessage(message DataFactoryMessage) {
	// 这里可以实现消息广播逻辑，比如通过 WebSocket 或消息队列
}

type DataFactoryUser struct {
	ID          uint            `gorm:"primaryKey"`
	Username    string          `gorm:"column:username;type:varchar(255);uniqueIndex"` // 用户名不加密
	Email       string          `gorm:"column:email"`                                  // 邮箱不加密
	Telephone   string          `gorm:"column:telephone"`                              // 邮箱不加密
	Role        string          `gorm:"column:role;default:user"`
	LoginMethod string          `gorm:"column:loginmethod;default:local"`
	Password    EncryptedString `gorm:"column:password"`
	GroupName   string          `gorm:"column:groupname;default:guess"`
	Profile     string          `gorm:"column:profile"`
	Extra       string          `gorm:"column:extra"`
	Activation  string          `gorm:"column:activation"`
	Expire      string          `gorm:"column:expire"`
	Checksum    string          `gorm:"column:checksum"`
	Timestamp   time.Time       `gorm:"column:timestamp"`
}

func (u *DataFactoryUser) BeforeSave(tx *gorm.DB) error {
	// 计算用户名的哈希值
	data := []byte(fmt.Sprintf("%s%s%s%s", u.Username, u.Role, u.GroupName, u.Password))
	u.Checksum = calculateChecksum(data)
	u.Timestamp = time.Now()
	return nil
}

// AfterFind 在读取用户数据后验证校验和
func (u *DataFactoryUser) AfterFind(tx *gorm.DB) error {
	data := []byte(fmt.Sprintf("%s%s%s%s", u.Username, u.Role, u.GroupName, u.Password))
	checksum := calculateChecksum(data)
	if checksum != u.Checksum {
		return errors.New("data integrity check failed")
	}
	return nil
}

func calculateHash(input string) string {
	// 创建一个 SHA256 哈希对象
	hasher := sha256.New()

	// 将输入字符串写入哈希对象
	hasher.Write([]byte(input))

	// 计算哈希值并返回十六进制字符串
	hashBytes := hasher.Sum(nil)
	return hex.EncodeToString(hashBytes)
}

func generateKey(username, email string) []byte {
	// 拼接用户名和邮箱
	key := []byte(username + email)

	// 如果密钥不足32字节，用'm'填充；如果超过32字节，截断
	if len(key) < 32 {
		key = append(key, bytes.Repeat([]byte{'m'}, 32-len(key))...)
	} else if len(key) > 32 {
		key = key[:32]
	}

	return key
}

func createDEEngine(cfg *ini.File) (*gorm.DB, error) {
	// 优先尝试从dataexpress节获取配置
	deCfg := cfg.Section("dataexpress")

	var (
		user   string
		passwd string
		host   string
		port   string
		dbname string
		key    *ini.Key
	)

	// 配置优先级判断
	if deCfg != nil { // 检查配置节是否存在
		if key = deCfg.Key("user"); key == nil || key.String() == "" {
			return nil, errors.New("数据库用户名未配置")
		}
		user = key.String()

		if key = deCfg.Key("password"); key == nil || key.String() == "" {
			return nil, errors.New("数据库密码未配置")
		}
		passwd = key.String()

		if key = deCfg.Key("host"); key == nil || key.String() == "" {
			return nil, errors.New("数据库主机地址未配置")
		}
		host = key.String()

		if key = deCfg.Key("port"); key == nil || key.String() == "" {
			port = "3306" // 设置默认端口
		} else {
			port = key.String()
		}

		if key = deCfg.Key("dbname"); key == nil || key.String() == "" {
			return nil, errors.New("数据库名未配置")
		}
		dbname = key.String()
	} else {
		return nil, errors.New("dataexpress配置节不存在")
	}

	// 构建DSN连接字符串
	mysqlDSN := fmt.Sprintf("%s:%s@tcp(%s:%s)/%s?charset=utf8&parseTime=True&loc=Local",
		user, passwd, host, port, dbname)

	// 建立数据库连接
	db, err := gorm.Open(mysql.Open(mysqlDSN), &gorm.Config{})
	if err != nil {
		return nil, fmt.Errorf("数据库连接失败: %v", err)
	}

	// 测试数据库连接
	sqlDB, err := db.DB()
	if err != nil {
		return nil, fmt.Errorf("获取数据库实例失败: %v", err)
	}
	if err := sqlDB.Ping(); err != nil {
		return nil, fmt.Errorf("数据库连接测试失败: %v", err)
	}

	log.Println("数据库连接成功")
	return db, nil
}

func initDB(host, port, user, passwd, dbname string) (*gorm.DB, error) {
	var err error
	var db *gorm.DB

	//mysqlDSN string =
	// 如果有 MySQL 配置，则使用 MySQL
	var mysqlDSN string
	if host != "" {
		mysqlDSN = fmt.Sprintf("%s:%s@tcp(%s:%s)/%s?charset=utf8&parseTime=True&loc=Local", user, passwd, host, port, dbname)
	} else {
		mysqlDSN = ""
	}

	if mysqlDSN != "" {
		db, err = gorm.Open(mysql.Open(mysqlDSN), &gorm.Config{})
		if err != nil {
			return nil, fmt.Errorf("无法连接 MySQL 数据库: %v", err)
		}
		log.Println("已连接到 MySQL 数据库")
	} else {
		// 使用 SQLite（注意：这里可以使用 sqlite3 或 glebarez/sqlite）
		db, err = gorm.Open(sqlite.Open("datafactory.db"), &gorm.Config{})
		if err != nil {
			return nil, fmt.Errorf("无法连接 SQLite 数据库: %v", err)
		}
		logger.Println("已连接到 SQLite 数据库")
	}

	// 自动迁移表结构
	if err := db.AutoMigrate(&DataFactoryFetchDataConfig{}); err != nil {
		return nil, fmt.Errorf("自动迁移表结构失败: %v", err)
	}

	// 自动迁移表结构
	if err := db.AutoMigrate(&DataFactoryUser{}); err != nil {
		return nil, fmt.Errorf("自动迁移表结构失败: %v", err)
	}
	if err := db.AutoMigrate(&DfLayoutOption{}); err != nil {
		log.Fatal("Failed to migrate database:", err)
	}

	if err := db.AutoMigrate(&DataFactoryMenuItem{}); err != nil {
		return nil, fmt.Errorf("自动迁移表结构失败: %v", err)
	}

	// 自动迁移表结构
	if err := db.AutoMigrate(&DataFactorySSOConfig{}); err != nil {
		return nil, fmt.Errorf("自动迁移表结构失败: %v", err)
	}
	// 自动迁移表结构
	if err := db.AutoMigrate(&DataFetchTask{}); err != nil {
		return nil, fmt.Errorf("自动迁移DataFetchTask表结构失败: %v", err)
	}

	if err := db.AutoMigrate(&DataFetchTable{}); err != nil {
		return nil, fmt.Errorf("自动迁移DataFetchTable表结构失败: %v", err)
	}
	if err := db.AutoMigrate(&InferFetchResult{}); err != nil {
		return nil, fmt.Errorf("自动迁移InferFetchResult表结构失败: %v", err)
	}

	if err := db.AutoMigrate(&UserSelfConfiguration{}); err != nil {
		return nil, fmt.Errorf("自动迁移UserSelfConfiguration表结构失败: %v", err)
	}

	if err := db.AutoMigrate(&POperationWithUsers{}); err != nil {
		return nil, fmt.Errorf("自动迁移POperationWithUsers表结构失败: %v", err)
	}

	if err := db.AutoMigrate(&PDataTask{}); err != nil {
		return nil, fmt.Errorf("自动迁移PDataTask表结构失败: %v", err)
	}

    if err := db.AutoMigrate(&PReportTemplate{}); err != nil {
		return nil, fmt.Errorf("自动迁移ReportTemplate表结构失败: %v", err)
	}
    if err := db.AutoMigrate(&ReportGenerationHistory{}); err != nil {
            return nil, fmt.Errorf("自动迁移ReportTemplate表结构失败: %v", err)
    }

	// 初始化管理员账号
	admins := []DataFactoryUser{
		{
			Username:    "data_admin",
			Role:        "admin",
			LoginMethod: "password",
			Password:    EncryptedString("data_admin"),
			GroupName:   "deeplearning",
		},
		{

			Username:    "assist",
			Role:        "admin",
			LoginMethod: "password",
			Password:    EncryptedString("assist"),
			GroupName:   "deeplearning",
		},
		{

			Username:    "user1",
			Role:        "user",
			LoginMethod: "password",
			Password:    EncryptedString("user1"),
			GroupName:   "deeplearning",
		},
	}
	for _, admin := range admins {
		var existingAdmin DataFactoryUser

		username := admin.Username
		//usernameHash := calculateHash(string(username))
		if err := db.Where("username =?", username).First(&existingAdmin).Error; err == nil {
			continue
		} else if !errors.Is(err, gorm.ErrRecordNotFound) {
			panic(fmt.Sprintf("failed to check admin existence: %v", err))
		}
		if err := db.Create(&admin).Error; err != nil {
			panic(fmt.Sprintf("failed to create admin user: %v", err))
		}
	}
	return db, nil
}

func loadDefaultData(file_path string) map[string]interface{} {
	// 先初始化为空map
	defaultData := make(map[string]interface{})

	file, err := os.ReadFile(file_path)
	if err != nil {
		log.Println("Warning: Failed to read options.json:", err)
		return defaultData // 返回空map
	}
	file = bytes.TrimPrefix(file, []byte{0xEF, 0xBB, 0xBF})
	// 尝试解析JSON
	if err := json.Unmarshal(file, &defaultData); err != nil {
		log.Println("Warning: Failed to parse options.json:", err)
		return make(map[string]interface{}) // 解析失败也返回空map
	}

	return defaultData
}

var err error

// spaFileServer 是一个自定义的静态文件处理器，支持 SPA 路由
func spaFileServer(root string) gin.HandlerFunc {
	fileSystem := http.Dir(root)
	fileServer := http.FileServer(fileSystem)

	return func(c *gin.Context) {
		// 检查请求的文件是否存在
		path := filepath.Join(root, c.Request.URL.Path)
		_, err := os.Stat(path)
		if os.IsNotExist(err) {
			// 如果文件不存在，返回 index.html
			c.File(filepath.Join(root, "index.html"))
			return
		} else if err != nil {
			// 其他错误
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Internal Server Error"})
			return
		}

		// 如果文件存在，返回文件
		fileServer.ServeHTTP(c.Writer, c.Request)
	}
}

type UserSelfConfiguration struct {
	ID         int    `json:"id"`
	UserID     string `json:"userId"`
	ConfigType string `json:"configType"` // 配置类型，如"theme"
	Value      string `json:"value"`      // 配置值，如主题名称
}

func saveUserSelfConfiguration(c *gin.Context) {
	var config UserSelfConfiguration

	// 从请求中获取配置数据
	if err := c.ShouldBindJSON(&config); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// 获取当前登录用户ID（从authMiddleware中设置）
	user, ok := getCurrentUser(c) //currentUser.(User)
	if !ok {
		logger.Warn("Failed to assert current user type")
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "内部服务器错误"})
		return
	}
	username := user.Username // 或从JWT中获取

	// 检查是否已存在该类型的配置
	var existingConfig UserSelfConfiguration
	result := db.Where("user_id = ? AND config_type = ?", username, config.ConfigType).First(&existingConfig)

	if result.Error == nil {
		// 存在则更新
		existingConfig.Value = config.Value
		db.Save(&existingConfig)
		c.JSON(http.StatusOK, gin.H{"message": "配置已更新", "data": existingConfig})
	} else {
		// 不存在则创建
		config.UserID = username
		db.Create(&config)
		c.JSON(http.StatusOK, gin.H{"message": "配置已保存", "data": config})
	}
}

// 获取配置
func getUserSelfConfiguration(c *gin.Context) {

	configType := c.Param("type")
	if configType == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "请指定配置类型"})
		return
	}
	log.Printf("type：%s", configType)

	// 获取当前登录用户ID
	user, ok := getCurrentUser(c) //currentUser.(User)
	if !ok {
		log.Printf("Failed to assert current user type")

		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "内部服务器错误"})
		return
	}
	log.Printf("current user,%s", user.Username)
	username := user.Username // 或从JWT中获取
	// 查询用户的配置
	var config UserSelfConfiguration
	result := db.Where("user_id = ? AND config_type = ?", username, configType).First(&config)

	if result.Error != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "未找到配置"})
		return
	}

	c.JSON(http.StatusOK, config)
}

//go:embed all:web/build
//go:embed all:config
//go:embed datafactory_fetchdata.py
//go:embed genCustomReport.py

var resources embed.FS // 用 embed.FS 类型的变量接收嵌入的资源
// 全局变量：资源释放目录
var releaseDir string

func main() {

	releaseDir := "." //filepath.Join(".", dirName)
	if err := releaseResources(releaseDir); err != nil {
		fmt.Printf("资源释放失败：%v，程序退出\n", err)
		return
	}
	fmt.Printf("资源已释放到：%s\n", releaseDir)
	// 3. 执行你的业务逻辑（示例）
	fmt.Println("开始执行业务逻辑...")

	cfg, err = ini.Load("config.ini")
	if err != nil {
		if os.IsNotExist(err) {
			cfg = ini.Empty()
		} else {
			log.Fatalf("无法加载配置文件: %v", err)
		}
	}
	mysqlSection := cfg.Section("mysql")
	if mysqlSection == nil {
		fmt.Println("MySQL section not found in the configuration file.")
		db, err = initDB("", "", "", "", "")
	} else {
		fmt.Println("user MySQL section config.ini init")
		host := mysqlSection.Key("host").String()
		port := mysqlSection.Key("port").String()
		user := mysqlSection.Key("user").String()
		passwd := mysqlSection.Key("passwd").String()
		dbname := mysqlSection.Key("dbname").String()
		db, err = initDB(host, port, user, passwd, dbname)
	}
	tmp, err := createDEEngine(cfg)
	if err != nil {
		log.Printf("创建主数据库引擎失败，使用备用连接: %v", err)
		de_engine = db // 假设 db 是预先定义好的备用连接
	} else {
		de_engine = tmp
	}
	// defaultOptionsData = loadDefaultData("options.json")
	// log.Printf("defaultOptionsData: %+v", defaultOptionsData)

	localip, err := getLocalIP()
	if err != nil {
		localip = "localhost"
		log.Printf("Failed to get local IP, using localhost: %v", err)
	}

	runMode := flag.String("mode", "run", "run/dev")
	run_mode = *runMode

	api_ip := flag.String("api_ip", localip, "ip to run the server on")

	port := flag.String("port", "5000", "Port to run the server on")
	log.Print("Server is running on port %s", port)

	flag.Parse()

	if err != nil {
		logger.Fatalf("数据库初始化失败: %v", err)
	}

	r := gin.Default()

	// 允许跨域
	r.Use(func(c *gin.Context) {
		origin := c.Request.Header.Get("Origin")
		c.Writer.Header().Set("Access-Control-Allow-Origin", origin) // "http://localhost:3000" 允许前端地址
		c.Writer.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		c.Writer.Header().Set("Access-Control-Allow-Credentials", "true") // 允许携带凭证
		if c.Request.Method == "OPTIONS" {
			log.Println("Handling OPTIONS request") // 调试日志
			c.AbortWithStatus(204)
			return
		}
		log.Printf("Handling %s request for %s\n", c.Request.Method, c.Request.URL.Path) // 调试日志
		c.Next()
	})

	// 创建数据目录
	if err := os.MkdirAll("data", 0755); err != nil {
		log.Fatal("Failed to create data directory:", err)
	}
	// 文件管理
	//     r.GET("/files", listFilesHandler)
	//     r.POST("/upload", uploadFileHandler)
	//     r.POST("/analyze", handleAnalysis)

	// 文件下载接口（隐藏实际路径）
	r.GET("/download/*filepath", func(c *gin.Context) { //authMiddleware,
		filePath := c.Param("filepath") // 文件唯一标识符

		log.Printf("Requested URL: %s", c.Request.URL.Path)
		log.Printf("Requested file path: %s", filePath)

		// 拼接实际文件路径
		fullPath := filepath.Join(".", filePath)
		log.Printf("Full file path: %s", fullPath)

		// 检查文件是否存在
		if _, err := os.Stat(fullPath); os.IsNotExist(err) {
			log.Printf("File not found: %s", fullPath)
			c.String(http.StatusNotFound, "File not found")
			return
		}

		// 提取文件名（路径的最后一部分）
		fileName := filepath.Base(fullPath)
		log.Printf("Extracted file name: %s", fileName)

		// 设置响应头，强制浏览器下载文件，并使用提取的文件名
		c.Header("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s\"", fileName))
		c.Header("Content-Type", "application/octet-stream")

		// 提供文件下载
		c.File(fullPath)
		log.Printf("File served: %s", fileName)
	})

	// API路由
	r.POST("/api/login", login)
	r.POST("/api/logout", logout)
	r.GET("/api/check-login", authMiddleware, checkLogin)
	r.GET("/api/auth/check", authMiddleware, checkLogin)

	r.GET("/api/fetchweb_layout_config", getConfig)
	r.GET("/api/fetchweb_layout_config_list", getConfigList) // 获取所有可用模板列表
	r.GET("/api/check-configname", checkConfigName)
	if run_mode == "dev" {
		r.GET("/api/configurations", getConfigurations)
	} else {
		r.GET("/api/configurations", authMiddleware, getConfigurations)
	}

	r.POST("/api/saveconfigurations", authMiddleware, saveConfiguration)
	r.GET("/api/configurations/:name", getConfigurationByID)
	r.POST("/api/user-selfconfig", authMiddleware, saveUserSelfConfiguration)
	r.GET("/api/user-selfconfig/:type", authMiddleware, getUserSelfConfiguration)

	r.DELETE("/api/configurations/:id", deleteConfigurationByID)
	r.POST("/api/fetch-data", fetchData)
	r.GET("/api/fetch-history", getFetchHistory)
	r.POST("/api/infer-fetch-data", inferFetchData)
	r.DELETE("/api/fetch-history/:taskId", deleteFetchTask) // 新增删除单个任务接口
	r.DELETE("/api/fetch-history", clearFetchHistory)       // 新增清空所有任务接口

	r.POST("/api/options/:category/:field", getOptions)
	// SSO 配置
	r.GET("/api/sso-config", getSSOConfig)
	r.POST("/api/sso-config", setSSOConfig)

	r.POST("/api/fetchdatatask-config-query", QueryConfigNameHandler)

	r.GET("/api/layouts", getLayouts)
	r.POST("/api/layouts", saveLayout)
	r.PUT("/api/layouts/:name", saveLayout)
	r.DELETE("/api/layouts/:name", deleteLayout)

	// 在您现有的路由初始化代码中添加
	menuGroup := r.Group("/api/menu-items")
	{
		menuGroup.GET("", getMenuItems)
		menuGroup.POST("", createMenuItem)
		menuGroup.PUT("/:key", updateMenuItem)
		menuGroup.DELETE("/:key", deleteMenuItem)
	}

	r.GET("/analysis/:configname", handleConfigAnalysis)
	r.POST("/analysis/upload", handleFileUpload)
	r.POST("/analysis/url", handleUrlAnalysis) // 新增URL分析路由
	r.POST("/analysis/advanced", handleAdvancedAnalysis)
	r.POST("/api/infer-history", QueryInferHistory)

	// 增删改查核心接口
	r.POST("/api/operations", createOperation)        // 创建操作
	r.GET("/api/operations", getOperations)           // 获取所有操作
	r.PUT("/api/operations/:key", updateOperation)    // 更新操作
	r.DELETE("/api/operations/:key", deleteOperation) // 删除操作

    r.POST("/api/report-template/save",authMiddleware, saveReportTemplate)
	r.GET("/api/report-template/list",authMiddleware, getReportTemplateList)

    r.POST("/api/report-template/gen_html",authMiddleware, generateHTMLHandler )

	RegisterBIRoutes(r)
	api := r.Group("/api")
	registerDataFetchRoutes(api)
	PDataSetupRoutes(api)

	buildDir := "build"

	r.NoRoute(func(c *gin.Context) {
		http.FileServer(http.Dir("build")).ServeHTTP(c.Writer, c.Request) //"build"
	})
	r.Use(spaFileServer(buildDir))

	startPort, err := strconv.Atoi(*port)
	if err != nil {
		log.Fatalf("无效的端口号: %v", err)
	}

	originalPort := startPort // 保存原始端口用于比较

	// 检查初始端口是否可用
	if !isPortAvailable(startPort) {
		fmt.Printf("端口 %d 已被占用\n", startPort)

		fmt.Print("自动尝试下一个可用端口?")

		// 寻找可用端口
		startPort = findAvailablePort(startPort)
		if startPort == -1 {
			log.Fatal("在范围内未找到可用端口，程序退出")
		}

		// 当找到的端口与原始端口不同时，需要用户确认
		if startPort != originalPort {
			fmt.Printf("\n找到可用端口: %d (与初始端口 %d 不同)\n", startPort, originalPort)
			fmt.Print("是否使用此端口启动服务? (按回车确认，输入其他字符取消): ")
			scanner := bufio.NewScanner(os.Stdin)
			scanner.Scan()
			confirm := scanner.Text()

			// 只有用户输入空字符串（直接回车）才确认
			if confirm != "" {
				fmt.Println("用户取消启动，程序退出")
				os.Exit(1)
			}
		}
	}

	*port = strconv.Itoa(startPort)
	// 启动服务器
	addr := fmt.Sprintf(":%s", *port)
	fmt.Printf("Server is running on port %s  \n", *port)
	go func() {
		time.Sleep(1 * time.Second) // 等待服务器启动
		//openBrowser(fmt.Sprintf("http://%s:%s", localip, *port))

		if err := openBrowser(fmt.Sprintf("http://%s:%s", localip, *port)); err != nil {
			log.Printf("Failed to open browser: %v\n", err)
			// 可以继续尝试其他方式，比如打印访问链接
			log.Printf("You can manually visit: http://%s:%s\n", localip, *port)
		}
	}()

	server_ip = *api_ip
	server_port = *port
	updateConfigIfNeeded(*api_ip, *port, buildDir)
	r.Run(addr) //":8005"

}

// 尝试监听端口，判断是否被占用
func isPortAvailable(port int) bool {
	listener, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return false
	}
	defer listener.Close()
	return true
}

// 自动寻找可用端口，从指定端口开始递增
func findAvailablePort(startPort int) int {
	port := startPort
	for {
		if isPortAvailable(port) {
			return port
		}
		port++
		// 防止无限循环，设置一个合理的端口范围上限
		if port > startPort+100 {
			return -1
		}
	}
}

// 获取 SSO 配置
func getSSOConfig(c *gin.Context) {
	var ssoConfigs []DataFactorySSOConfig // 使用切片来存储查询结果
	if err := db.Find(&ssoConfigs).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// 如果没有记录，返回空数组 []
	if len(ssoConfigs) == 0 {
		c.JSON(http.StatusOK, []DataFactorySSOConfig{}) // 返回空数组
		return
	}

	// 返回查询到的第一条记录（假设只需要一条记录）
	c.JSON(http.StatusOK, ssoConfigs[0])
}

func getConfigList(c *gin.Context) {
	// 查询数据库中的所有模板
	var layouts []DfLayoutOption
	if err := db.Find(&layouts).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "数据库查询失败"})
		return
	}

	// 构造返回的模板列表（包含 default + 数据库里的模板）
	result := []gin.H{
		{"name": "Demo", "description": "仅供参考，实际模板需管理员配置"},
	}

	for _, layout := range layouts {
		result = append(result, gin.H{
			"name": layout.LayoutName,
			//             "description": layout.Description, // 如果有 description 字段
		})
	}

	c.JSON(http.StatusOK, result)
}

func getMenuItems(c *gin.Context) {

	var items []DataFactoryMenuItem

	// 按照Order字段排序获取菜单项
	if err := db.Order("\"order\" asc").Find(&items).Error; err != nil {
		log.Printf("数据库查询失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "获取菜单项失败",
		})
		return
	}

	c.JSON(http.StatusOK, items)
}

// 辅助函数：获取请求体内容
func getRequestBody(c *gin.Context) string {
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		return fmt.Sprintf("无法读取请求体: %v", err)
	}
	// 恢复请求体以便后续处理
	c.Request.Body = io.NopCloser(bytes.NewBuffer(body))
	return string(body)
}

func createMenuItem(c *gin.Context) {
	log.Printf("开始处理创建菜单项请求 - 客户端IP: %s", c.ClientIP())

	var item DataFactoryMenuItem
	if err := c.ShouldBindJSON(&item); err != nil {
		log.Printf("请求参数绑定失败: %v - 请求体: %s", err, getRequestBody(c))
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "无效的请求参数",
			"details": err.Error(),
		})
		return
	}

	log.Printf("接收到菜单项数据: %+v", item)
	// 检查Key是否已存在
	var count int64
	if err := db.Model(&DataFactoryMenuItem{}).Where("itemkey = ?", item.Itemkey).Count(&count).Error; err != nil {
		log.Printf("检查菜单项Key存在性失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "服务器内部错误",
		})
		return
	}

	if count > 0 {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "菜单项Key已存在",
		})
		return
	}

	// 创建菜单项
	if err := db.Create(&item).Error; err != nil {
		log.Printf("创建菜单项失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "创建菜单项失败",
		})
		return
	}

	log.Printf("菜单项创建成功，ID: %d", item.ID)

	c.JSON(http.StatusOK, gin.H{
		"status":  "success",
		"message": "菜单项创建成功",
		"data":    item,
	})
}

func updateMenuItem(c *gin.Context) {
	// 获取路径参数中的key
	id := c.Param("key")
	if id == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "菜单项key不能为空",
		})
		return
	}

	var item DataFactoryMenuItem
	if err := c.ShouldBindJSON(&item); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "无效的请求参数",
			"details": err.Error(),
		})
		return
	}

	// 检查菜单项是否存在 - 修复查询条件，使用Itemkey字段
	var existingItem DataFactoryMenuItem
	if err := db.Where("itemkey = ?", id).First(&existingItem).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			c.JSON(http.StatusNotFound, gin.H{
				"error": "菜单项未找到",
			})
		} else {
			log.Printf("查询菜单项失败: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "服务器内部错误",
			})
		}
		return
	}

	// 更新菜单项 - 确保使用路径中的key
	item.Itemkey = id
	// 保留原始ID，避免更新时修改主键
	item.ID = existingItem.ID

	if err := db.Save(&item).Error; err != nil {
		log.Printf("更新菜单项失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "更新菜单项失败",
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":  "success",
		"message": "菜单项更新成功",
		"data":    item,
	})
}

func deleteMenuItem(c *gin.Context) {
	// 修复变量未定义问题，添加变量声明
	key := c.Param("key")
	if key == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "菜单项key不能为空",
		})
		return
	}

	// 检查菜单项是否存在 - 修复查询条件，使用Itemkey字段
	var item DataFactoryMenuItem
	if err := db.Where("itemkey = ?", key).First(&item).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			c.JSON(http.StatusNotFound, gin.H{
				"error": "菜单项未找到",
			})
		} else {
			log.Printf("查询菜单项失败: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "服务器内部错误",
			})
		}
		return
	}

	// 删除菜单项
	if err := db.Delete(&item).Error; err != nil {
		log.Printf("删除菜单项失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "删除菜单项失败",
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":  "success",
		"message": "菜单项删除成功",
	})
}

// 设置 SSO 配置
func setSSOConfig(c *gin.Context) {
	var config DataFactorySSOConfig
	if err := c.ShouldBindJSON(&config); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	// 如果时间戳是零值，设置为当前时间
	if config.Timestamp.IsZero() {
		config.Timestamp = time.Now()
	}

	// 清空原有配置并保存新配置
	db.Where("1 = 1").Delete(&DataFactorySSOConfig{})
	if err := db.Create(&config).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, config)
}

type DfLayoutOption struct {
	ID          uint            `gorm:"primaryKey"`
	LayoutName  string          `gorm:"column:layout_name;type:varchar(255);uniqueIndex"`
	LayoutJSON  json.RawMessage `gorm:"type:json"`
	OptionsJSON json.RawMessage `gorm:"type:json"`
	UpdatedAt   time.Time       `gorm:"autoUpdateTime"`
}

type RequestData struct {
	LayoutName string   `json:layout_name`
	Field      string   `json:"field"`
	Level      string   `json:"level"`
	Tab        string   `json:"tab"`
	Depends    []string `json:"depends"`
}

func getMapKeys(m map[string]interface{}) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

// 处理全局选项
func handleGlobalOptions(globalData map[string]interface{}, field string, dependsOn []string) interface{} {
	if len(dependsOn) == 0 {
		// 无依赖项的情况
		val, exists := globalData[field]
		if !exists {
			return []string{}
		}
		// 确保返回列表形式
		if slice, ok := val.([]interface{}); ok {
			return slice
		}
		return []interface{}{val}
	}

	// 多依赖项处理
	fieldData, ok := globalData[field]
	if !ok {
		return []string{}
	}

	return processDependentValues(fieldData, dependsOn)
}

// 核心：处理多依赖项合并去重
func processDependentValues(fieldData interface{}, dependsOn []string) []string {
	resultSet := make(map[string]struct{})

	fieldMap, ok := fieldData.(map[string]interface{})
	if !ok {
		return []string{}
	}

	for _, dependValue := range dependsOn {
		val, exists := fieldMap[dependValue]
		if !exists {
			continue
		}

		switch v := val.(type) {
		case []interface{}:
			for _, item := range v {
				if str, ok := item.(string); ok {
					resultSet[str] = struct{}{}
				}
			}
		case []string:
			for _, item := range v {
				resultSet[item] = struct{}{}
			}
		case string:
			resultSet[v] = struct{}{}
		}
	}

	result := make([]string, 0, len(resultSet))
	for k := range resultSet {
		result = append(result, k)
	}
	return result
}

func getCategoryData(optionsData map[string]interface{}, category string) (map[string]interface{}, error) {
	rawData, exists := optionsData[category]
	if !exists {
		return nil, fmt.Errorf("category '%s' not found", category)
	}

	// 尝试解析为 map[string]interface{}
	if data, ok := rawData.(map[string]interface{}); ok {
		return data, nil
	}

	// 尝试解析为 map[string][]string
	if data, ok := rawData.(map[string][]string); ok {
		result := make(map[string]interface{})
		for k, v := range data {
			result[k] = v
		}
		return result, nil
	}

	return nil, fmt.Errorf("unsupported type for category '%s': %T", category, rawData)
}

func getDependentValues(fieldData interface{}, dependsOn []string) []string {
	resultSet := make(map[string]struct{})

	if fieldMap, ok := fieldData.(map[string]interface{}); ok {
		for _, dependValue := range dependsOn {
			if val, exists := fieldMap[dependValue]; exists {
				switch v := val.(type) {
				case []interface{}:
					for _, item := range v {
						if str, ok := item.(string); ok {
							resultSet[str] = struct{}{}
						}
					}
				case []string:
					for _, item := range v {
						resultSet[item] = struct{}{}
					}
				case string:
					resultSet[v] = struct{}{}
				}
			}
		}
	}

	result := make([]string, 0, len(resultSet))
	for k := range resultSet {
		result = append(result, k)
	}
	return result
}
func parseOptionsJSON(raw json.RawMessage) (map[string]interface{}, error) {
	data := make(map[string]interface{})

	// 尝试直接解析
	if err := json.Unmarshal(raw, &data); err == nil {
		return data, nil
	}

	// 尝试去除 BOM 头
	clean := bytes.TrimPrefix(raw, []byte{0xEF, 0xBB, 0xBF})
	if err := json.Unmarshal(clean, &data); err == nil {
		return data, nil
	}

	// 尝试解析为字符串化的 JSON
	var str string
	if err := json.Unmarshal(raw, &str); err == nil {
		if err := json.Unmarshal([]byte(str), &data); err == nil {
			return data, nil
		}
	}

	return nil, fmt.Errorf("all parsing attempts failed")
}

func getOptions(c *gin.Context) {
	// 获取路径参数
	logger.Info("post /api/options/:category/:field getOptions")
	var requestData struct {
		LayoutName string   `json:"layout_name"`
		Field      string   `json:"field"`
		Level      string   `json:"level"`
		Tab        string   `json:"tab"`
		Depends    []string `json:"depends"`
	}

	if err := c.ShouldBindJSON(&requestData); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"code":    400,
			"message": "请求体必须为JSON格式",
		})
		return
	}

	log.Printf("Query params: %+v", c.Request.URL.Query())
	log.Printf("Request body: %+v", requestData)

	layoutName := requestData.LayoutName
	if layoutName == "" {
		layoutName = "default"
	}

	category := c.Param("category")
	field := c.Param("field")
	dependsOn := requestData.Depends

	// 1. 尝试从数据库获取指定布局
	var DfLayoutOption DfLayoutOption
	err := db.Where("layout_name = ?", layoutName).First(&DfLayoutOption).Error

	var optionsData map[string]interface{}

	// 数据加载逻辑保持不变
	if layoutName != "default" {
		optionsData, err = parseOptionsJSON(DfLayoutOption.OptionsJSON)
		if err != nil {
			log.Println("parseOptionsJSON error:", err)
			// optionsData = defaultOptionsData

		} else {
			log.Println("db optionjson parse succ")
		}
	} else {
		log.Printf("user defaultOptionsData to optionsData: %s", defaultOptionsData)
		// optionsData = defaultOptionsData
	}
	log.Printf("optionsData: %+v", optionsData)
	log.Printf("category[%s] field[%s]", category, field)
	log.Printf("request_data: %+v", requestData)

	defer func() {
		if r := recover(); r != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"code":    500,
				"message": "服务器内部错误",
			})
		}
	}()

	var data interface{}
	log.Printf("optionsData: %+v", optionsData)

	if category == "global" {
		globalData := optionsData["global"].(map[string]interface{})
		log.Printf("globalData: %+v  dependsOn %+v", globalData, dependsOn)

		if len(dependsOn) > 0 {
			data = processMultiDepends(globalData[field], dependsOn)
		} else {
			// 无依赖项逻辑保持不变
			val, exists := globalData[field]
			if !exists {
				data = []interface{}{}
			} else if _, isSlice := val.([]interface{}); !isSlice && val != nil {
				data = []interface{}{val}
			} else {
				data = val
			}
		}
	} else {
		// 其他分类选项处理
		if _, exists := optionsData[category]; !exists {
			log.Printf("category '%s' not found in optionsData", category)
			c.JSON(http.StatusOK, []string{})
			return
		}

		categoryData, err := getCategoryData(optionsData, category)
		if err != nil {
			log.Printf("Error: %v", err)
			c.JSON(http.StatusOK, []string{})
			return
		}

		fieldData, ok := categoryData[field]
		if !ok {
			log.Printf("categoryData[%s] ", field)
			log.Printf("categoryData:%s", categoryData)
			c.JSON(http.StatusOK, []string{})
			return
		}

		if len(dependsOn) > 0 {
			data = processMultiDepends(fieldData, dependsOn)
		} else {
			if fieldMap, ok := fieldData.(map[string][]string); ok {
				keys := make([]string, 0, len(fieldMap))
				for k := range fieldMap {
					keys = append(keys, k)
				}
				data = keys
			} else {
				data = fieldData
			}
		}
	}

	log.Printf("return option data: %+v", data)
	c.JSON(http.StatusOK, data)
}

// 新增核心处理函数
func processMultiDepends(fieldData interface{}, dependsOn []string) interface{} {
	resultSet := make(map[string]struct{})

	fieldMap, ok := fieldData.(map[string]interface{})
	if !ok {
		return []string{}
	}

	for _, dependValue := range dependsOn {
		val, exists := fieldMap[dependValue]
		if !exists {
			continue
		}

		switch v := val.(type) {
		case []interface{}:
			for _, item := range v {
				if str, ok := item.(string); ok {
					resultSet[str] = struct{}{}
				}
			}
		case []string:
			for _, item := range v {
				resultSet[item] = struct{}{}
			}
		case string:
			resultSet[v] = struct{}{}
		}
	}

	result := make([]string, 0, len(resultSet))
	for k := range resultSet {
		result = append(result, k)
	}
	return result
}

func openBrowser(url string) error {
	var commands map[string][]string
	switch runtime.GOOS {
	case "linux":
		commands = map[string][]string{
			"xdg-open":      {url},
			"x-www-browser": {url},
			"gnome-open":    {url},
			"kde-open":      {url},
		}
	case "windows":
		commands = map[string][]string{
			"rundll32": {"url.dll,FileProtocolHandler", url},
			"start":    {url}, // Windows 10+ 支持
		}
	case "darwin":
		commands = map[string][]string{
			"open": {url},
		}
	default:
		return fmt.Errorf("unsupported platform")
	}

	for cmd, args := range commands {
		err := exec.Command(cmd, args...).Start()
		if err == nil {
			return nil // 成功则直接返回
		}
	}
	return fmt.Errorf("no compatible browser command found")
}

// 获取配置
func getConfig(c *gin.Context) {
	templateName := c.Query("template")

	// 默认模板
	if templateName == "" || templateName == "default" {
		configFile := "./fetch_data_config.json"
		data, err := os.ReadFile(configFile)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "默认配置加载失败"})
			return
		}

		// 尝试解析JSON，确保格式正确
		var parsedJSON interface{}
		if err := json.Unmarshal(data, &parsedJSON); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "默认配置格式错误"})
			return
		}

		c.JSON(http.StatusOK, parsedJSON)
		return
	}

	// 数据库模板
	var layout DfLayoutOption
	if err := db.Where("layout_name = ?", templateName).First(&layout).Error; err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "模板不存在"})
		return
	}

	// 确保LayoutJSON是有效的JSON
	var parsedJSON interface{}
	if err := json.Unmarshal(layout.LayoutJSON, &parsedJSON); err != nil {
		// 如果直接解析失败，尝试先去除可能的引号
		stripped := strings.Trim(string(layout.LayoutJSON), "\"")
		if err := json.Unmarshal([]byte(stripped), &parsedJSON); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "模板解析失败: " + err.Error()})
			return
		}
	}
	c.JSON(http.StatusOK, parsedJSON)
}

// 检查配置名称是否存在
func checkConfigName(c *gin.Context) {
	configName := c.Query("name")
	if configName == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "参数不能为空"})
		return
	}

	var exists bool
	err := db.Model(&DataFactoryFetchDataConfig{}).
		Select("count(*) > 0").
		Where("configname = ?", configName).
		Find(&exists).
		Error

	if err != nil {
		return
	}

	c.JSON(http.StatusOK, gin.H{"exists": exists})
}

// 定义一个包含完整配置和任务信息的结构体
type ConfigWithTask struct {
	Configname   string         `json:"configname"`
	Timerange    string         `json:"timerange"`
	Dates        datatypes.JSON `json:"dates"`
	Selections   datatypes.JSON `json:"selections"`
	Detail       datatypes.JSON `json:"detail"`
	Savedat      string         `json:"savedat"`
	Layoutconfig datatypes.JSON `json:"layoutconfig"`
	Modifier     string         `json:"modifier"`
	Type         string         `json:"type"`
	LayoutName   string         `json:"layout_name"`
	// 任务字段 - 保持与SQL别名一致
	TaskStatus        string     `json:"status"`
	TaskTriggeredat   time.Time  `json:"triggeredat" `
	TaskStatusdetails string     `json:"statusdetails" `
	TaskDownloadinfo  string     `json:"downloadinfo" `
	TaskCompletedat   *time.Time `json:"completedat,omitempty" `
	SqlJson           string     `json:"sql_json"`
}

func getConfigurations(c *gin.Context) {
	logger.Info("get /api/configurations getConfigurations")

	// 首先获取所有配置，并按 savedat 降序排序（最新的在前面）
	var configs []ConfigWithTask
	// 添加 Order("savedat DESC") 实现按保存时间降序排列
	if run_mode == "dev" {
		if err := db.Table("data_factory_fetch_data_configs").Order("savedat DESC").Find(&configs).Error; err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "获取配置失败"})
			return
		}
	} else {
		user, ok := getCurrentUser(c) //currentUser.(User)
		if !ok {
			logger.Warn("Failed to assert current user type")
			c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "内部服务器错误"})
			return
		}
		username := user.Username // 或从JWT中获取
		if err := db.Table("data_factory_fetch_data_configs").
			Where("modifier = ?", username). // 匹配 Modifier 字段
			Order("savedat DESC").
			Find(&configs).Error; err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "获取配置失败"})
			return
		}
	}

	// 然后为每个配置获取最新的任务
	configNames := make([]string, len(configs))
	for i, config := range configs {
		configNames[i] = config.Configname
	}

	// 获取每个配置的最新任务
	var latestTasks []struct {
		Configname    string
		Status        string
		Triggeredat   time.Time
		Statusdetails string
		Downloadinfo  string
		Completedat   time.Time
	}

	// 使用子查询先找出每个配置的最新触发时间
	subQuery := db.Table("data_fetch_tasks").
		Select("configname, MAX(triggeredat) as latest_triggeredat").
		Where("configname IN (?)", configNames).
		Group("configname")

	// 然后关联回原表获取完整记录
	if err := db.Table("data_fetch_tasks as t1").
		Select("t1.configname, t1.status, t1.triggeredat, t1.statusdetails, t1.downloadinfo, t1.completedat").
		Joins("JOIN (?) as t2 ON t1.configname = t2.configname AND t1.triggeredat = t2.latest_triggeredat", subQuery).
		Find(&latestTasks).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取任务状态失败"})
		return
	}

	// 将任务映射到配置
	taskMap := make(map[string]interface{})
	for _, task := range latestTasks {
		taskMap[task.Configname] = map[string]interface{}{
			"status":        task.Status,
			"triggeredAt":   task.Triggeredat,
			"statusDetails": task.Statusdetails,
			"downloadInfo":  task.Downloadinfo,
			"completedAt":   task.Completedat,
		}
	}

	// 构建响应（保持查询时的排序）
	response := make([]map[string]interface{}, len(configs))
	for i, config := range configs {
		response[i] = map[string]interface{}{
			"configname":   config.Configname,
			"timerange":    config.Timerange,
			"dates":        config.Dates,
			"selections":   config.Selections,
			"detail":       config.Detail,
			"savedat":      config.Savedat,
			"layoutconfig": config.Layoutconfig,
			"modifier":     config.Modifier,
			"type":         config.Type,
			"recentTask":   taskMap[config.Configname],
			"layout_name":  config.LayoutName,
			"sql_json":     config.SqlJson,
		}
	}

	c.JSON(http.StatusOK, response)
}

// 保存配置
func saveConfiguration(c *gin.Context) {
	var config DataFactoryFetchDataConfig
	if err := c.ShouldBindJSON(&config); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "无效的请求数据"})
		return
	}
	log.Printf("config: %s", config)
	username1 := c.GetString("username") // 或从JWT中获取
	log.Printf("username: %s", username1)
	// 获取当前用户信息（假设从请求头或上下文中获取）

	user, ok := getCurrentUser(c) //currentUser.(User)
	if !ok {
		logger.Warn("Failed to assert current user type")
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "内部服务器错误"})
		return
	}
	username := user.Username // 或从JWT中获取
	log.Printf("current_user: %s", username)
	if username == "" {
		username = "system" // 默认用户名
	}

	// 设置保存时间和修改者
	config.Savedat = time.Now().Format("2006-01-02 15:04:05")
	config.Modifier = username // 假设你在结构体中添加了Modifier字段
	if len(config.SqlJson) > 0 {
		config.Type = "sql"
	} else {
		config.Type = "normal"
	}
	// 检查配置是否已存在
	var existingConfig DataFactoryFetchDataConfig
	result := db.Where("configname = ?", config.Configname).First(&existingConfig)

	if errors.Is(result.Error, gorm.ErrRecordNotFound) {
		if config.Type == "sql" && len(config.SqlJson) == 0 {
			log.Printf("config type == sql and sqljson len is 0")
			c.JSON(http.StatusBadRequest, gin.H{
				"error":   "SQL配置保存失败",
				"details": "类型为SQL的配置必须包含有效的sql_json字段",
			})
			return
		}
		// 配置不存在，创建新配置
		result = db.Create(&config)
		if result.Error != nil {
			log.Printf("create error:  %s ", result.Error.Error())
			c.JSON(http.StatusInternalServerError, gin.H{"error": "保存配置失败", "details": result.Error.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"message":      "配置保存成功",
			"configname":   config.Configname,
			"type":         config.Type, // 返回配置类型
			"rowsAffected": result.RowsAffected,
			"modifier":     config.Modifier,
			"layout_name":  config.LayoutName,
		})
	} else {
		// 配置已存在，更新现有配置
		// 只更新非零值字段
		updateData := map[string]interface{}{
			"timerange":    config.Timerange,
			"dates":        config.Dates,
			"selections":   config.Selections,
			"detail":       config.Detail,
			"savedat":      config.Savedat,
			"layoutconfig": config.Layoutconfig,
			"modifier":     config.Modifier,
			"type":         config.Type, // 更新配置类型
		}

		if config.Type == "sql" {
			// SQL配置必须包含 SqlJson
			if len(config.SqlJson) == 0 {
				c.JSON(http.StatusBadRequest, gin.H{
					"error":   "SQL配置更新失败",
					"details": "类型为SQL的配置必须包含有效的sql_json字段",
				})
				return
			}
			updateData["sql_json"] = config.SqlJson // SQL配置更新 sql_json
		} else {
			updateData["detail"] = config.Detail // 普通配置更新 detail
		}

		result = db.Model(&existingConfig).Updates(updateData)
		if result.Error != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "更新配置失败:" + result.Error.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"message":      "配置更新成功",
			"configname":   config.Configname,
			"rowsAffected": result.RowsAffected,
		})
	}
}

// 根据ID获取配置
func getConfigurationByID(c *gin.Context) {
	logger.Info("get /api/configurations/:id getConfigurationByID")

	// 获取路径参数中的id
	name := c.Param("name")
	if name == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "配置name不能为空"})
		return
	}

	// 首先获取指定ID的配置，并按savedat降序排序
	var configs []ConfigWithTask
	query := db.Table("data_factory_fetch_data_configs").Where("configname = ?", name) // 添加ID过滤

	if err := query.Order("savedat DESC").Find(&configs).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取配置失败"})
		return
	}

	// 如果没有找到配置
	if len(configs) == 0 {
		c.JSON(http.StatusNotFound, gin.H{"error": "未找到指定配置"})
		return
	}

	// 复用原有的获取最新任务逻辑
	configNames := make([]string, len(configs))
	for i, config := range configs {
		configNames[i] = config.Configname
	}

	var latestTasks []struct {
		Configname    string
		Status        string
		Triggeredat   time.Time
		Statusdetails string
		Downloadinfo  string
		Completedat   time.Time
	}

	subQuery := db.Table("data_fetch_tasks").
		Select("configname, MAX(triggeredat) as latest_triggeredat").
		Where("configname IN (?)", configNames).
		Group("configname")

	if err := db.Table("data_fetch_tasks as t1").
		Select("t1.configname, t1.status, t1.triggeredat, t1.statusdetails, t1.downloadinfo, t1.completedat").
		Joins("JOIN (?) as t2 ON t1.configname = t2.configname AND t1.triggeredat = t2.latest_triggeredat", subQuery).
		Find(&latestTasks).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取任务状态失败"})
		return
	}

	taskMap := make(map[string]interface{})
	for _, task := range latestTasks {
		taskMap[task.Configname] = map[string]interface{}{
			"status":        task.Status,
			"triggeredAt":   task.Triggeredat,
			"statusDetails": task.Statusdetails,
			"downloadInfo":  task.Downloadinfo,
			"completedAt":   task.Completedat,
		}
	}

	// 构建响应（因为是按ID查询，结果应该只有一个）
	response := make([]map[string]interface{}, len(configs))
	for i, config := range configs {
		response[i] = map[string]interface{}{
			"configname":   config.Configname,
			"timerange":    config.Timerange,
			"dates":        config.Dates,
			"selections":   config.Selections,
			"detail":       config.Detail,
			"savedat":      config.Savedat,
			"layoutconfig": config.Layoutconfig,
			"modifier":     config.Modifier,
			"type":         config.Type,
			"recentTask":   taskMap[config.Configname],
			"layout_name":  config.LayoutName,
			"sql_json":     config.SqlJson,
		}
	}

	// 如果只需要返回单个对象而非数组，可以直接取第一个元素
	if len(response) > 0 {
		c.JSON(http.StatusOK, response)
	} else {
		c.JSON(http.StatusNotFound, gin.H{"error": "未找到指定配置"})
	}
}

// 根据ID删除配置
func deleteConfigurationByID(c *gin.Context) {
	// 获取配置名称参数
	id := c.Param("id")

	// 直接删除配置，不需要先查询
	result := db.Where("configname = ?", id).Delete(&DataFactoryFetchDataConfig{})

	if result.Error != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "删除配置失败", "details": result.Error.Error()})
		return
	}

	if result.RowsAffected == 0 {
		c.JSON(http.StatusNotFound, gin.H{"error": "配置不存在"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message":      "配置删除成功",
		"configname":   id,
		"rowsAffected": result.RowsAffected,
	})
}

// 数据采集请求结构体
type DataFetchRequest struct {
	ConfigName     string          `json:"configName"`
	TimeRange      string          `json:"timeRange"`
	Dates          interface{}     `json:"dates"`
	Configurations json.RawMessage `json:"configurations"`
	LayoutConfig   json.RawMessage `json:"layoutconfig"`
	SqlJson        json.RawMessage `json:"sql_json"`    // 新增：接收SQL配置
	Description    string          `json:"description"` // 新增：任务描述
}

func logRequestInfo(c *gin.Context, request interface{}) {
	log.Printf("请求方法: %s", c.Request.Method)
	log.Printf("请求URL: %s", c.Request.URL.String())
	log.Printf("请求来源IP: %s", c.ClientIP())

	requestData, _ := json.Marshal(request)
	log.Printf("请求参数: %s", string(requestData))
}

// 转换函数
func convertToMapString(data interface{}) (map[string]string, bool) {
	result := make(map[string]string)

	// 检查data是否为map[string]interface{}类型
	if dataMap, ok := data.(map[string]interface{}); ok {
		for k, v := range dataMap {
			// 将每个值转换为字符串
			if strVal, ok := v.(string); ok {
				result[k] = strVal
			} else {
				// 尝试其他类型转换
				strVal := fmt.Sprintf("%v", v)
				result[k] = strVal
			}
		}
		return result, true
	}

	// 检查data是否为[]byte类型（JSON原始数据）
	if dataBytes, ok := data.([]byte); ok {
		var dataMap map[string]interface{}
		if err := json.Unmarshal(dataBytes, &dataMap); err != nil {
			log.Printf("❌ 解析日期配置失败: %v", err)
			return nil, false
		}
		for k, v := range dataMap {
			if strVal, ok := v.(string); ok {
				result[k] = strVal
			} else {
				strVal := fmt.Sprintf("%v", v)
				result[k] = strVal
			}
		}
		return result, true
	}

	// 检查data是否为string类型（JSON字符串）
	if dataStr, ok := data.(string); ok {
		var dataMap map[string]interface{}
		if err := json.Unmarshal([]byte(dataStr), &dataMap); err != nil {
			log.Printf("❌ 解析日期配置失败: %v", err)
			return nil, false
		}
		for k, v := range dataMap {
			if strVal, ok := v.(string); ok {
				result[k] = strVal
			} else {
				strVal := fmt.Sprintf("%v", v)
				result[k] = strVal
			}
		}
		return result, true
	}

	log.Printf("❌ 日期配置类型不支持: %T", data)
	return nil, false
}

// normalizeJSON 将任意JSON数据转换为有序的map结构，用于比较
func normalizeJSON(data interface{}) (interface{}, error) {
	var jsonData []byte
	var err error

	// 处理不同类型的输入
	switch v := data.(type) {
	case json.RawMessage:
		jsonData = []byte(v)
	case string:
		jsonData = []byte(v)
	case []byte:
		jsonData = v
	default:
		// 尝试序列化其他类型
		jsonData, err = json.Marshal(v)
		if err != nil {
			return nil, fmt.Errorf("无法序列化数据: %v", err)
		}
	}

	// 反序列化为通用结构
	var normalized interface{}
	err = json.Unmarshal(jsonData, &normalized)
	if err != nil {
		return nil, fmt.Errorf("JSON解析失败: %v", err)
	}

	// 递归处理嵌套结构，确保数组顺序一致
	return normalizeValue(normalized), nil
}

// normalizeValue 递归处理JSON值，确保数组有序
func normalizeValue(v interface{}) interface{} {
	switch val := v.(type) {
	case map[string]interface{}:
		// 处理对象
		for k, v2 := range val {
			val[k] = normalizeValue(v2)
		}
		return val
	case []interface{}:
		// 处理数组
		for i, v2 := range val {
			val[i] = normalizeValue(v2)
		}
		return val
	default:
		// 基本类型直接返回
		return val
	}
}

// 数据采集
func fetchData(c *gin.Context) {
	log.Println("========== 开始处理数据采集请求 ==========")

	var request DataFetchRequest

	// 记录请求信息
	logRequestInfo(c, request)

	if err := c.ShouldBindJSON(&request); err != nil {
		log.Printf("❌ 请求数据解析失败: %v", err)

		// 尝试获取更详细的错误信息
		if syntaxError, ok := err.(*json.SyntaxError); ok {
			log.Printf("JSON语法错误在位置: %d", syntaxError.Offset)
		}

		c.JSON(http.StatusBadRequest, gin.H{
			"success": false,
			"error":   "无效的请求数据: " + err.Error(),
		})
		return
	}
	log.Printf("✅ 请求解析成功, 配置名称: %s, 时间范围类型: %s,配置内容大小: %d 字节, 布局配置大小: %d 字节 , SqlJson:%d字节",
		request.ConfigName, request.TimeRange, len(request.Configurations), len(request.LayoutConfig), len(request.SqlJson))
	log.Printf("日期配置: %+v", request.Dates)

	// 记录请求类型（普通配置/ SQL配置）
	configType := "normal"
	if len(request.SqlJson) > 0 {
		configType = "sql"
		log.Printf("✅ 检测到SQL类型配置，配置名称: %s", request.ConfigName)
	} else {
		log.Printf("✅ 检测到普通类型配置，配置名称: %s", request.ConfigName)
	}

	// 将 request.Dates 序列化为 JSON 字符串
	datesBytes, err := json.Marshal(request.Dates)
	if err != nil {
		log.Printf("❌ 序列化 Dates 失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"success": false,
			"error":   "序列化 Dates 失败: " + err.Error(),
		})
		return
	}

	log.Printf("开始检测，取数配置是否在取数配置表中保存")
	// 检查是否存在该配置名称的保存配置
	var savedConfig DataFactoryFetchDataConfig
	result := db.Where("configname = ?", request.ConfigName).First(&savedConfig)

	// 检查查询结果
	if result.Error != nil {
		if errors.Is(result.Error, gorm.ErrRecordNotFound) {
			// 明确记录不存在的情况
			log.Printf("❌ 未找到配置: %s", request.ConfigName)
			c.JSON(http.StatusOK, gin.H{
				"success": false,
				"error":   "配置未入库，请先保存配置",
			})
			return
		} else {
			// 其他错误情况
			log.Printf("❌ 查询配置失败: %v", result.Error)
			c.JSON(http.StatusOK, gin.H{
				"success": false,
				"error":   "检查配置失败",
			})
			return
		}
	}

	// 如果查询成功，继续处理
	log.Printf("✅ 找到配置: %s", request.ConfigName)
	// 配置类型校验
	if savedConfig.Type != configType {
		log.Printf("❌ 配置类型不匹配，请求类型: %s, 存储类型: %s", configType, savedConfig.Type)
		c.JSON(http.StatusOK, gin.H{
			"success": false,
			"error":   "配置类型不匹配，请重新保存配置",
		})
		return
	}

	log.Printf("开始检测，取数配置 是和表中存储的不一致 ")
	// 比较当前配置与已保存配置是否相同
	var errorMsg string
	if configType == "sql" {

		// SQL配置检查逻辑
		reqSqlJson, err := normalizeJSON(request.SqlJson)
		savedSqlJson, err := normalizeJSON(savedConfig.SqlJson)

		if err != nil {
			errorMsg = "SQL配置解析失败"
		} else if !reflect.DeepEqual(reqSqlJson, savedSqlJson) {
			errorMsg = "SQL配置与已存不一致，请先保存"
		}
	} else {
		// 比较各字段
		if request.TimeRange != savedConfig.Timerange {
			errorMsg = "时间范围类型与已存不一致，请先保存"
		} else {
			// 将请求中的Dates转换为有序结构进行比较
			_, err := normalizeJSON(request.Dates) //reqDatesMap
			if err != nil {
				errorMsg = "日期配置解析失败"
				log.Printf("❌ 请求Dates解析失败: %v", err)
			} else {
				// 将数据库中的Dates转换为有序结构
				_, err := normalizeJSON(savedConfig.Dates) //savedDatesMap
				if err != nil {
					errorMsg = "已保存日期配置解析失败"
					log.Printf("❌ 已保存Dates解析失败: %v", err)
				}

			}
		}

		// 比较 Configurations 和 Detail
		if errorMsg == "" {
			reqConfig, err := normalizeJSON(request.Configurations)     //
			savedConfigDetail, err := normalizeJSON(savedConfig.Detail) //
			if err != nil {
				errorMsg = "配置解析失败"
				log.Printf("❌ Configurations 解析失败: %v", err)
			} else if !reflect.DeepEqual(reqConfig, savedConfigDetail) {
				log.Printf("fetchData Configurations 取数配置和表中存储不一致 ")
				errorMsg = "此项目取数配置与数据库保存不一致，请先保存更新"
			}
		}

		// 比较 LayoutConfig 和 Layoutconfig
		if errorMsg == "" {
			reqLayout, err := normalizeJSON(request.LayoutConfig)
			savedLayout, err := normalizeJSON(savedConfig.Layoutconfig)
			if err != nil {
				errorMsg = "布局配置解析失败"
				log.Printf("❌ LayoutConfig 解析失败: %v", err)
			} else if !reflect.DeepEqual(reqLayout, savedLayout) {
				errorMsg = "此项目请求Layout配置与数据库保存不一致，请先保存进行更新"
			}
		}

		if errorMsg != "" {
			log.Printf("取数配置和表中存储不一致,需保存 ")
			c.JSON(http.StatusOK, gin.H{
				"success": false,
				"error":   fmt.Sprintf("配置已更改，请先保存配置：%s", errorMsg),
				"message": "取数任务已提交",
			})
			return
		}
	}

	// 	datesString := string(datesBytes)
	// 创建任务记录
	taskID := fmt.Sprintf("task%s", time.Now().Format("060102150405.000"))
	taskID = strings.ReplaceAll(taskID, ".", "")
	log.Printf("生成任务ID: %s", taskID)

	task := DataFetchTask{
		ID:            taskID,
		ConfigName:    request.ConfigName,
		Configuration: string(request.Configurations),
		TimeRange:     request.TimeRange,
		Dates:         string(datesBytes),
		LayoutConfig:  string(request.LayoutConfig),
		SqlJson:       datatypes.JSON(request.SqlJson), // 新增：存储SQL配置
		Description:   request.Description,             // 新增：任务描述
		Type:          configType,                      // 新增：任务类型
		TriggeredAt:   time.Now(),
		Status:        "pending",
		StatusDetails: "任务已创建，等待处理",
	}

	if err := db.Create(&task).Error; err != nil {
		log.Printf("❌ 任务创建失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"success": false,
			"error":   "任务创建失败: " + err.Error(),
		})
		return
	}
	log.Printf("✅ 任务记录保存到数据库成功")

	log.Printf("配置内容大小: %d 字节, 布局配置大小: %d 字节", len(request.Configurations), len(request.LayoutConfig))
	updateResult := db.Model(&DataFetchTask{}).Where("id = ?", taskID).Updates(map[string]interface{}{
		"status":        "processing",
		"statusdetails": "正在处理数据请求",
	})

	if updateResult.Error != nil {
		log.Printf("❌ 更新任务状态为处理中失败: %v", updateResult.Error)
		return
	}

	if updateResult.RowsAffected == 0 {
		log.Printf("⚠️ 更新任务状态为处理中失败，未找到任务记录: %s", taskID)
		return
	}
	log.Printf("✅ 任务状态已更新为 processing")

	currentTime := time.Now().Format("200601021020")
	log.Printf("当前时间格式: %s", currentTime)
	// 启动异步任务处理
	// 序列化请求数据为JSON
	requestJSON, err := json.Marshal(request)
	if err != nil {
		log.Printf("❌ JSON序列化失败 (任务: %s): %v", taskID, err)
		updateTaskStatus(taskID, "failed", "数据处理准备失败")
		return
	}
	if err := os.MkdirAll("taskconfig", 0755); err != nil {
		log.Printf("❌Failed to create data directory : %v", taskID, err)
		return
	}
	// 创建临时文件保存配置
	configPath := fmt.Sprintf("./taskconfig/%s.json", taskID)
	tmpFile, err := os.Create(configPath) // 使用 Create 而不是 CreateTemp
	if err != nil {
		log.Printf("❌ 创建配置文件失败 (任务: %s): %v", taskID, err)
		updateTaskStatus(taskID, "failed", "数据处理准备失败")
		return
	}

	//defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.Write(requestJSON); err != nil {
		log.Printf("❌ 写入临时文件失败 (任务: %s): %v", taskID, err)
		updateTaskStatus(taskID, "failed", "数据处理准备失败")
		return
	}
	path_name := tmpFile.Name()
	tmpFile.Close()

	var emptyInferRequest InferFetchBaseTask
	runFetchData(configType, taskID, path_name, request, emptyInferRequest, FetchModeTrain)

	log.Printf("返回响应给前端")
	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"taskId":  task.ID,
		"message": "取数任务已提交",
	})
	log.Printf("========== 结束处理 取数任务 %s ==========", taskID)
}

const (
	FetchModeTrain = "train" // 训练模式
	FetchModeInfer = "infer" // 推理模式
)

type ProcessResult struct {
	Data  interface{} // 推理结果（dataframe解析后的数据）
	Error error       // 错误信息
}

func runFetchData(
	configType string,
	taskID string,
	configPath string,
	request DataFetchRequest,
	inferTask InferFetchBaseTask,
	fetchMode string,
) *ProcessResult {
	// 校验模式合法性
	if fetchMode != FetchModeTrain && fetchMode != FetchModeInfer {
		log.Printf("❌ 无效的取数模式: %s，任务ID: %s", fetchMode, taskID)
		err := fmt.Errorf("无效的取数模式: %s", fetchMode)
		return &ProcessResult{Error: err}
	}

	// 根据配置类型和模式调度处理器
	if fetchMode == FetchModeTrain {
		if configType == "sql" {
			if cfg.HasSection("DataExpress") {
				if !cfg.Section("DataExpress").HasKey("use_python_query") {
					// 走Python处理器（支持训练/推理）
					go runPythonDataProcessor(taskID, configPath, inferTask, fetchMode)
				} else {
					// 走SQL处理器（支持训练/推理）
					go runSqlDataProcessor(taskID, request, inferTask, fetchMode)
				}
			} else {
				// 无配置节时默认走SQL处理器
				go runSqlDataProcessor(taskID, request, inferTask, fetchMode)
			}
		} else {
			// 非SQL类型走Python处理器
			go runPythonDataProcessor(taskID, configPath, inferTask, fetchMode)
		}
		return nil // 训练模式无需返回结果

	} else {
		// resultCh := make(chan *ProcessResult, 1)
		var data interface{}
		if cfg.HasSection("DataExpress") {
			if !cfg.Section("DataExpress").HasKey("use_python_query") {
				// 走Python处理器（支持训练/推理）
				data, err = runPythonDataProcessor(taskID, configPath, inferTask, fetchMode)
			} else {
				// 走SQL处理器（支持训练/推理）
				data, err = runSqlDataProcessor(taskID, request, inferTask, fetchMode)
			}
		} else {
			// 无配置节时默认走SQL处理器
			data, err = runSqlDataProcessor(taskID, request, inferTask, fetchMode)
		}
		return &ProcessResult{Data: data, Error: err}

	}

}

// 启动Python脚本处理任务
func runPythonDataProcessor(taskID string, configPath string, inferTask InferFetchBaseTask, fetchMode string) (interface{}, error) {
	log.Printf("启动Python处理脚本（模式: %s），任务ID: %s", fetchMode, taskID)

	pythonCmd := "python3"
	// 检查python3是否存在
	if _, err := exec.LookPath(pythonCmd); err != nil {
		log.Printf("未找到python3，尝试使用python")
		pythonCmd = "python"
		// 检查python是否存在
		if _, err := exec.LookPath(pythonCmd); err != nil {
			log.Printf("未找到Python解释器  ,python3和python均不存在")
			updateTaskStatus(taskID, "failed", "未找到Python解释器")
			return nil, fmt.Errorf("未找到Python解释器")
		}
	}
	// 生成唯一的输出JSON文件路径（基于taskID，避免冲突）

	// 确保临时文件目录存在（如果需要）
	os.MkdirAll("tmp", 0755) // 可选，根据实际路径调整
	//outputJSONPath :=
	outputJSONPath := filepath.Join("data", fmt.Sprintf("tmp_download_info_%s.json", taskID))

	// 构建Python命令
	// cmd := exec.Command(
	// 	pythonCmd,
	// 	"datafactory_fetchdata.py",
	// 	"--task-id", taskID,
	// 	"--config-file", config_path,
	// 	"--mode", fetchMode, // 传递模式参数
	// )
	cmdArgs := []string{
		"datafactory_fetchdata.py",
		"--task-id", taskID,
		"--config-file", configPath,
		"--mode", fetchMode, // 传递模式参数
		"--output-json", outputJSONPath,
	}

	if fetchMode == FetchModeInfer {
		inferParamsJSON, err := json.Marshal(inferTask.SqlJson)
		if err != nil {
			log.Printf("❌ 序列化推理参数失败: %v", err)
			updateInferTaskStatus(taskID, "failed", "未找到Python解释器")
			return nil, fmt.Errorf("未找到Python解释器")
		}
		cmdArgs = append(cmdArgs, "--infer-params", string(inferParamsJSON)) // 传递推理参数
	}
	cmd := exec.Command(pythonCmd, cmdArgs...)

	// 捕获Python脚本的输出
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	log.Printf("执行Python脚本: %v", cmd.Args)

	// 执行命令
	if err := cmd.Run(); err != nil {
		log.Printf("❌ Python脚本执行失败 (任务: %s): %v\n%s (%s)", taskID, err, stderr.String(), stdout.String())
		updateTaskStatus(taskID, "failed", fmt.Sprintf("处理失败: %v %s", err, stderr.String()))
		return nil, fmt.Errorf("Python执行失败: %v, 错误日志: %s", err, stderr.String())
	}
	stdoutStr := string(stdout.Bytes())
	log.Printf("✅ Python脚本执行成功 (任务: %s): %s", taskID, stdoutStr)
	// 执行成功后，读取输出JSON文件获取download_info

	if fetchMode == FetchModeInfer {
		updateInferTaskStatus(taskID, "completed", "数据处理完成")
		var dataframe []map[string]interface{} // 适配pandas的orient="records"格式
		if err := json.Unmarshal(stdout.Bytes(), &dataframe); err != nil {
			return nil, fmt.Errorf("解析dataframe失败: %v, 原始输出: %s", err, stdout.String())
		}

		return dataframe, nil
	} else {

		var downloadInfo map[string]interface{}
		outputData, err := os.ReadFile(outputJSONPath)
		if err != nil {
			log.Printf("读取输出JSON文件失败: %v", err)
			updateTaskStatus(taskID, "completed", "获取下载信息失败")
			return nil, fmt.Errorf("读取下载信息失败: %v", err)
		}
		if err := json.Unmarshal(outputData, &downloadInfo); err != nil {
			log.Printf("解析输出JSON失败: %v, 原始内容: %s", err, string(outputData))
			updateTaskStatus(taskID, "failed", "解析下载信息失败")
			return nil, fmt.Errorf("解析下载信息失败: %v", err)
		}

		// 6. 更新任务状态为完成
		db.Model(&DataFetchTask{}).Where("id = ?", taskID).Updates(map[string]interface{}{
			"status":        "completed",
			"statusdetails": "Python取数成功",
			"downloadinfo":  string(outputData),
			"completedat":   time.Now(),
		})

		log.Printf("✅ 任务 %s 处理完成，结果文件: %s", taskID, downloadInfo["single_file"])

		updateTaskStatus(taskID, "completed", "数据处理完成")

		// 假设 outputJSONPath 是要删除的文件路径
		// terr := os.Remove(outputJSONPath)
		// if terr != nil {
		// 	// 处理错误，比如文件不存在或没有权限
		// 	log.Println("删除文件失败: %w", err)
		// }

		return nil, nil
	}
	//updateTaskStatus(taskID, "completed", "数据处理完成")
}

func runSqlDataProcessor(taskID string, request DataFetchRequest, inferTask InferFetchBaseTask, fetchmode string) (interface{}, error) {
	log.Printf("开始处理SQL取数任务: %s", taskID)

	if fetchmode == FetchModeTrain {

		// 1. 更新任务状态为"processing"
		db.Model(&DataFetchTask{}).Where("id = ?", taskID).Updates(map[string]interface{}{
			"status":        "processing",
			"statusdetails": "正在执行SQL查询",
		})
	}
	// 2. 解析SQL配置
	var sqlConfig map[string]interface{}
	var rawJsonStr string

	if fetchmode == FetchModeTrain {
		// 尝试解析外层JSON字符串（处理可能的双重编码）
		if err := json.Unmarshal([]byte(request.SqlJson), &rawJsonStr); err != nil {
			rawJsonStr = string(request.SqlJson)
		}
	} else {
		if err := json.Unmarshal([]byte(inferTask.SqlJson), &rawJsonStr); err != nil {
			rawJsonStr = string(inferTask.SqlJson)
		}
	}

	// 解析实际的SQL配置
	if err := json.Unmarshal([]byte(rawJsonStr), &sqlConfig); err != nil {
		log.Printf("❌ 解析SQL配置失败: %v; sqlconfig:%s", err, sqlConfig)
		errMsg := fmt.Sprintf("解析SQL配置失败: %v", err)
		updateTaskStatus(taskID, "failed", "解析SQL配置失败: "+err.Error())
		return nil, fmt.Errorf(errMsg)
	}

	// 3. 提取SQL和参数
	sql, ok := sqlConfig["sql"].(string)
	if !ok {
		errMsg := "SQL配置缺少sql字段"
		updateTaskStatus(taskID, "failed", errMsg)
		return nil, fmt.Errorf(errMsg)
	}

	params, _ := sqlConfig["params"].(map[string]interface{}) // 参数可选
	// configName, _ := sqlConfig["config_name"].(string)        // 配置名可选

	// 4. 检测SQL风险
	sqlRisk := CheckSqlRisk(sql)
	if sqlRisk.IsRisky {
		log.Printf("⚠️ SQL存在风险: %s", sqlRisk.Reason)

		// 对于危险操作可以直接阻止执行
		if sqlRisk.RiskLevel == RiskLevelDanger {
			errMsg := fmt.Sprintf("SQL包含危险操作: %s", sqlRisk.Reason)
			updateTaskStatus(taskID, "failed", errMsg)
			return nil, fmt.Errorf(errMsg)
		}

		// 对于警告级别的操作，可以记录日志但继续执行，或根据需要进行处理
		updateTaskStatus(taskID, "processing", "SQL包含需要注意的操作: "+sqlRisk.Reason)
	}

	// 4. 执行SQL查询
	df, err := executeSqlQuery(sql, params)
	if err != nil {
		log.Printf("❌ SQL查询执行失败: %v", err)
		errMsg := fmt.Sprintf("SQL查询执行失败: %v", err)
		updateTaskStatus(taskID, "failed", "SQL查询执行失败: "+err.Error())
		return nil, fmt.Errorf(errMsg)
	}

	if fetchmode == FetchModeInfer {
		return df, nil
	}

	// 5. 保存结果到CSV文件
	downloadInfo, err := saveSqlResultToFile(taskID, df)
	if err != nil {
		log.Printf("❌ 保存结果文件失败: %v", err)
		errMsg := fmt.Sprintf("保存结果文件失败: %v", err)
		updateTaskStatus(taskID, "failed", "保存结果文件失败: "+err.Error())
		return nil, fmt.Errorf(errMsg)
	}
	downloadInfoJSON, err := json.Marshal(downloadInfo)
	if err != nil {
		errMsg := fmt.Sprintf("转换下载信息为JSON失败: %v", err)
		log.Printf("❌ 转换下载信息为JSON失败: %v", err)
		updateTaskStatus(taskID, "failed", "生成下载信息失败")
		return nil, fmt.Errorf(errMsg)
	}

	// 6. 更新任务状态为完成
	db.Model(&DataFetchTask{}).Where("id = ?", taskID).Updates(map[string]interface{}{
		"status":        "completed",
		"statusdetails": "SQL取数成功",
		"downloadinfo":  string(downloadInfoJSON),
		"completedat":   time.Now(),
	})

	log.Printf("✅ 任务 %s 处理完成，结果文件: %s", taskID, downloadInfo["single_file"])
	return nil, nil
}

// 保存结果为CSV文件（返回下载信息）
func saveSqlResultToFile(taskID string, df *DataFrame) (map[string]interface{}, error) {
	// 创建数据目录
	dataDir := filepath.Join("data", "sql_results")
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return nil, fmt.Errorf("创建目录失败: %v", err)
	}

	// 生成文件名（格式：配置名_任务ID_时间戳.csv）
	// if configName == "" {
	// 	configName = "result"
	// }  , configName string
	fileName := fmt.Sprintf("%s.csv", taskID) // time.Now().UnixNano()
	filePath := filepath.Join(dataDir, fileName)

	// 保存为CSV
	if err := saveDataFrameToCSV(df, filePath); err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"single_file": "download/" + filePath, // 前端可访问的路径
	}, nil
}

// 执行SQL查询（使用预设的全局数据库连接）
func executeSqlQuery(sql string, params map[string]interface{}) (*DataFrame, error) {
	// 检查全局数据库连接是否有效
	if de_engine == nil {
		return nil, fmt.Errorf("全局数据库连接未初始化")
	}

	// 验证连接有效性
	sqlDB, err := de_engine.DB()
	if err != nil {
		return nil, fmt.Errorf("获取数据库连接失败: %v", err)
	}

	if err := sqlDB.Ping(); err != nil {
		return nil, fmt.Errorf("数据库连接不可用: %v", err)
	}

	// 替换SQL中的命名参数
	var queryArgs []interface{}
	paramNames := make([]string, 0, len(params))
	for name := range params {
		paramNames = append(paramNames, name)
	}

	sort.Slice(paramNames, func(i, j int) bool {
		return len(paramNames[i]) > len(paramNames[j])
	})

	for _, name := range paramNames {
		placeholder := fmt.Sprintf(":%s", name)
		if strings.Contains(sql, placeholder) {
			sql = strings.ReplaceAll(sql, placeholder, "?")
			queryArgs = append(queryArgs, params[name])
		}
	}

	// 执行查询
	log.Printf("执行SQL: %s, 参数: %v", sql, queryArgs)
	rows, err := de_engine.Raw(sql, queryArgs...).Rows()
	if err != nil {
		return nil, fmt.Errorf("SQL执行失败: %v (SQL: %s)", err, sql)
	}
	defer rows.Close()

	// 获取列信息
	columns, err := rows.Columns()
	if err != nil {
		return nil, fmt.Errorf("获取列信息失败: %v", err)
	}

	// 构建DataFrame
	df := &DataFrame{
		Columns: columns,
		Rows:    make([]map[string]interface{}, 0),
	}

	for rows.Next() {
		values := make([]interface{}, len(columns))
		valuePtrs := make([]interface{}, len(columns))
		for i := range columns {
			valuePtrs[i] = &values[i]
		}

		if err := rows.Scan(valuePtrs...); err != nil {
			return nil, fmt.Errorf("扫描行数据失败: %v", err)
		}

		row := make(map[string]interface{})
		for i, col := range columns {
			val := values[i]
			if val == nil {
				row[col] = nil
			} else if b, ok := val.([]byte); ok {
				row[col] = string(b)
			} else {
				row[col] = val
			}
		}
		df.Rows = append(df.Rows, row)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("行迭代错误: %v", err)
	}

	log.Printf("SQL查询成功，返回 %d 条数据", len(df.Rows))
	return df, nil
}

// DataFrame结构
type DataFrame struct {
	Columns []string
	Rows    []map[string]interface{}
}

// 保存DataFrame为CSV文件
func saveDataFrameToCSV(df *DataFrame, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("创建文件失败: %v", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// 写入表头
	if err := writer.Write(df.Columns); err != nil {
		return fmt.Errorf("写入表头失败: %v", err)
	}

	// 写入数据行
	for _, row := range df.Rows {
		record := make([]string, len(df.Columns))
		for i, col := range df.Columns {
			val := row[col]
			if val == nil {
				record[i] = ""
			} else {
				record[i] = fmt.Sprintf("%v", val) // 简单类型转换
			}
		}
		if err := writer.Write(record); err != nil {
			return fmt.Errorf("写入数据行失败: %v", err)
		}
	}

	return nil
}

// 更新任务状态的辅助函数
func updateTaskStatus(taskID, status, details string) {
	if err := db.Model(&DataFetchTask{}).
		Where("id = ?", taskID).
		Updates(map[string]interface{}{
			"status":        status,
			"statusdetails": details,
			"completedat":   time.Now(),
		}).Error; err != nil {

		log.Printf("更新任务状态失败 (任务: %s): %v", taskID, err)
	} else {
		log.Printf("更新任务状态: %s -> %s  %s ", taskID, status, details)
	}
}
func updateInferTaskStatus(recordID string, status string, details string, extraData ...map[string]interface{}) {
	updateData := map[string]interface{}{
		"status":        status,
		"statusdetails": details,
	}
	if status == "completed" || status == "failed" {
		updateData["end_time"] = time.Now() // 推理表用end_time字段
	}
	// 合并额外数据（如推理结果JSON）
	if len(extraData) > 0 && extraData[0] != nil {
		for k, v := range extraData[0] {
			updateData[k] = v
		}
	}
	// 更新推理结果表
	db.Model(&InferFetchResult{}).Where("id = ?", recordID).Updates(updateData)
}

type DataFetchTask struct {
	ID            string         `gorm:"primarykey;column:id" json:"id"`
	ConfigName    string         `gorm:"column:configname" json:"configname"`
	Type          string         `gorm:"column:type;type:varchar(50)" json:"type"`  // 新增：区分配置类型（normal/sql）
	TimeRange     string         `gorm:"column:timeRange" json:"timeRange"`         //`json:"timeRange"`
	Dates         string         `gorm:"column:dates" json:"dates"`                 // `json:"dates"`
	Configuration string         `gorm:"column:configuration" json:"configuration"` // 存为JSON字符串
	LayoutConfig  string         `gorm:"column:layoutconfig" json:"layoutconfig"`
	SqlJson       datatypes.JSON `gorm:"column:sql_json;type:json" json:"sql_json"`
	TriggeredAt   time.Time      `gorm:"column:triggeredat" json:"triggeredAt"`
	CompletedAt   *time.Time     `gorm:"column:completedat" json:"completedAt,omitempty"`
	Status        string         `gorm:"column:status" json:"status"` // pending, processing, success, failed
	StatusDetails string         `gorm:"column:statusdetails" json:"statusDetails,omitempty"`
	Description   string         `gorm:"column:description;type:text" json:"description"` // 新增描述字段
	DownloadInfo  string         `gorm:"column:downloadinfo" json:"downloadInfo,omitempty"`
}

// 获取历史任务
func getFetchHistory(c *gin.Context) {
	log.Println("========== 开始获取历史任务列表 ==========")

	// 从查询参数获取配置名称
	configName := c.Query("configName")
	if configName == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"success": false,
			"error":   "缺少配置名称参数",
		})
		return
	}

	log.Printf("正在查询配置 [%s] 的历史任务", configName)
	var tasks []DataFetchTask
	result := db.Where("configname = ?", configName).
		Order("triggeredat DESC").
		Limit(50).
		Find(&tasks)

	if result.Error != nil {
		log.Printf("❌ 查询历史任务失败: %v", result.Error)
		c.JSON(http.StatusInternalServerError, gin.H{
			"success": false,
			"error":   "查询历史任务失败",
		})
		return
	}

	log.Printf("✅ 查询到 %d 条历史任务记录", len(tasks))

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"tasks":   tasks,
	})

	log.Println("========== 完成获取历史任务列表 ==========")
}

func deleteFetchTask(c *gin.Context) {
	taskID := c.Param("taskId")

	// 校验任务是否存在
	var task DataFetchTask
	result := db.First(&task, "id = ?", taskID)
	if result.Error != nil {
		if errors.Is(result.Error, gorm.ErrRecordNotFound) {
			c.JSON(http.StatusNotFound, gin.H{"success": false, "error": "任务不存在"})
		} else {
			c.JSON(http.StatusInternalServerError, gin.H{"success": false, "error": "删除任务失败"})
		}
		return
	}

	// 删除任务
	if err := db.Delete(&task).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"success": false, "error": "删除任务失败"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"success": true, "message": "任务删除成功"})
}

func clearFetchHistory(c *gin.Context) {
	configName := c.Query("configName")
	if configName == "" {
		c.JSON(http.StatusBadRequest, gin.H{"success": false, "error": "缺少配置名称参数"})
		return
	}

	// 删除该配置的所有任务
	result := db.Where("configname = ?", configName).Delete(&DataFetchTask{})
	if result.Error != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"success": false, "error": "清空历史记录失败"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"message": fmt.Sprintf("成功清空 %d 条历史记录", result.RowsAffected),
	})
}

// validateSSOLogin SSO 登录验证
func validateSSOLogin(username, password string) error {
	// 查询 SSO 配置
	var ssoConfig DataFactorySSOConfig
	if err := db.First(&ssoConfig).Error; err != nil {
		return fmt.Errorf("failed to fetch SSO config: %v", err)
	}

	// 组织请求参数
	params := url.Values{}
	if ssoConfig.UrlEncode {
		params.Set(ssoConfig.UserField, url.QueryEscape(username))
		params.Set(ssoConfig.PasswordField, url.QueryEscape(password))
	} else {
		params.Set(ssoConfig.UserField, username)
		params.Set(ssoConfig.PasswordField, password)
	}

	// 发送请求
	var resp *http.Response
	var err error
	if ssoConfig.Method == "GET" {
		// GET 请求
		requestURL := fmt.Sprintf("%s?%s", ssoConfig.SSOUrl, params.Encode())
		log.Println(" requestURL : %s", requestURL)
		//log.info("url:%s", requestURL)
		resp, err = http.Get(requestURL)
	} else if ssoConfig.Method == "POST" {
		// POST 请求
		requestBody := params.Encode()
		resp, err = http.Post(ssoConfig.SSOUrl, "application/x-www-form-urlencoded", bytes.NewBufferString(requestBody))
	} else {
		return fmt.Errorf("invalid SSO method: %s", ssoConfig.Method)
	}

	if err != nil {
		return fmt.Errorf("failed to send SSO request: %v", err)
	}
	defer resp.Body.Close()

	// 读取响应
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read SSO response: %v", err)
	}
	log.Println("========== resultbody : %s", body)
	// 判断登录是否成功
	switch ssoConfig.SsoSussJudge {
	case `text is "0"`:
		if string(body) != "0" {
			return fmt.Errorf("SSO login failed: response is not '0'")
		}
	case `text is "succ"`:
		if string(body) != "succ" {
			return fmt.Errorf("SSO login failed: response is not 'succ'")
		}
	case "ret json contain token&not nul":
		var result map[string]interface{}
		if err := json.Unmarshal(body, &result); err != nil {
			return fmt.Errorf("failed to parse SSO response: %v", err)
		}
		if token, ok := result["token"].(string); !ok || token == "" {
			return fmt.Errorf("SSO login failed: token is missing or empty")
		}
	case "ret json contain code":
		var result map[string]interface{}
		if err := json.Unmarshal(body, &result); err != nil {
			return fmt.Errorf("failed to parse SSO response: %v", err)
		}
		if _, ok := result["code"]; !ok {
			return fmt.Errorf("SSO login failed: code is missing")
		}
	default:
		return fmt.Errorf("invalid SSO success judge: %s", ssoConfig.SsoSussJudge)
	}

	return nil
}

var jwtSecret = []byte("your-secret-key") // 从环境变量读取更安全

type Claims struct {
	Username string `json:"username"`
	Role     string `json:"role"`
	jwt.RegisteredClaims
}

// 生成Token
func GenerateToken(username, role string) (string, error) {
	claims := Claims{
		Username: username,
		Role:     role,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(24 * time.Hour)),
		},
	}
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString(jwtSecret)
}

// 解析Token
func ParseToken(tokenString string) (*Claims, error) {
	token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
		return jwtSecret, nil
	})
	if claims, ok := token.Claims.(*Claims); ok && token.Valid {
		return claims, nil
	}
	return nil, err
}

// 解密激活码
func decryptActivationCode(activationCode string, key []byte) (int64, error) {
	// Base64解码
	ciphertext, err := base64.StdEncoding.DecodeString(activationCode)
	if err != nil {
		return 0, err
	}

	// 创建AES解密器
	block, err := aes.NewCipher(key)
	if err != nil {
		return 0, err
	}

	if len(ciphertext) < aes.BlockSize {
		return 0, errors.New("ciphertext too short")
	}

	// 解密数据
	decryptedData := make([]byte, len(ciphertext))
	mode := cipher.NewCBCDecrypter(block, make([]byte, aes.BlockSize))
	mode.CryptBlocks(decryptedData, ciphertext)

	// 去除填充
	decryptedData = unpad(decryptedData)

	// 解析过期时间戳
	var expirationTimestamp int64
	_, err = fmt.Sscanf(string(decryptedData), "%d", &expirationTimestamp)
	if err != nil {
		return 0, err
	}

	return expirationTimestamp, nil
}

// 去除填充
func unpad(data []byte) []byte {
	length := len(data)
	unpadding := int(data[length-1])
	return data[:(length - unpadding)]
}

// 验证激活码
func validateActivationCode(username, email, activationCode string) error {
	// 生成密钥
	key := generateKey(username, email)

	// 解密激活码
	expirationTimestamp, err := decryptActivationCode(activationCode, key)
	if err != nil {
		return fmt.Errorf("failed to decrypt activation code: %v", err)
	}

	// 检查激活码是否过期
	if time.Now().Unix() > expirationTimestamp {
		return errors.New("激活码已过期，请重新激活或购买授权")
	}

	// 打印过期时间
	expirationTime := time.Unix(expirationTimestamp, 0)
	fmt.Printf("激活码过期时间: %s\n", expirationTime.Format("2006-01-02 15:04:05"))

	return nil
}

// 从激活码中提取过期时间
func extractExpirationTime(activationCode string) (int64, error) {
	// 这里假设激活码是Base64编码的加密字符串
	// 解密逻辑与之前相同
	key := generateKey("username", "email") // 根据实际情况生成密钥
	expirationTimestamp, err := decryptActivationCode(activationCode, key)
	if err != nil {
		return 0, err
	}
	return expirationTimestamp, nil
}

var loggedInUsers = make(map[string]bool) // 全局已登录用户存储

func valid_db_password(db *gorm.DB, username, password string, n int) (*DataFactoryUser, error) {
	var user DataFactoryUser

	// 使用 GORM 查询用户
	result := db.Where("username = ?", username).First(&user)
	if result.Error != nil {
		if errors.Is(result.Error, gorm.ErrRecordNotFound) {
			return nil, errors.New("用户不存在")
		}
		return nil, result.Error
	}

	// 验证密码
	if string(user.Password) != password {
		return nil, errors.New("密码错误")
	}

	// // 获取当前用户数量
	// count, err := getCurrentUserCount(db)
	// if err != nil {
	// 	return nil, err
	// }

	// 判断是否需要验证激活码
	if user.ID > uint(n) {
		// 验证激活码是否过期
		expirationTimestamp, err := extractExpirationTime(user.Activation)
		if err != nil {
			return nil, errors.New("激活码无效")
		}
		if time.Now().Unix() > expirationTimestamp {
			return nil, errors.New("激活码已过期")
		}
	}

	return &user, nil
}

func setCookie(c *gin.Context, key, value string, maxAge int, secure, httpOnly bool) {
	domain := c.Request.Host
	if strings.Contains(domain, ":") {
		domain = strings.Split(domain, ":")[0]
	}

	// 如果是IP或localhost，不设置Domain
	var cookieDomain string
	if !isIPAddress(domain) && domain != "localhost" {
		cookieDomain = domain
	}

	c.SetCookie(
		key,
		value,
		maxAge,
		"/",
		cookieDomain,
		secure,
		httpOnly,
	)
}

// 辅助函数：检查是否是IP地址
// func isIPAddress(addr string) bool {
//     return net.ParseIP(addr) != nil
// }

// login 登录函数
func login(c *gin.Context) {
	logger.Info("Entered login handler")

	var req struct {
		Username string `json:"username"`
		Password string `json:"password"`
	}

	// 解析请求体
	if err := c.ShouldBindJSON(&req); err != nil {
		logger.Errorf("Failed to bind JSON: %v", err)
		c.JSON(http.StatusBadRequest, gin.H{"status": "error", "message": "无效的请求"})
		return
	}

	logger.Infof("req username: %s", req.Username)
	// 查询用户
	var user DataFactoryUser
	if err := db.Where("username = ?", req.Username).First(&user).Error; err != nil {
		logger.Warnf("User not found in database: %s, trying SSO as fallback", req.Username)
		// 数据库未找到用户，尝试使用SSO验证
		if err := validateSSOLogin(req.Username, req.Password); err != nil {
			logger.Warnf("SSO login failed for username: %s, error: %v", req.Username, err)
			c.JSON(http.StatusUnauthorized, gin.H{"status": "error", "message": "用户名或密码错误"})
			return
		}
		// SSO验证成功，创建临时用户
		user.Username = req.Username
		user.Role = "user" // 默认角色，可以根据SSO信息调整
		user.LoginMethod = "sso"
	} else {
		// 根据 loginMethod 选择验证方式
		switch user.LoginMethod {
		case "password":
		case "local":
			// 本地验证
			if _, err := valid_db_password(db, req.Username, req.Password, 1303); err != nil {
				logger.Warnf("Login failed for username: %s, error: %v", req.Username, err)
				c.JSON(http.StatusUnauthorized, gin.H{"status": "error", "message": err.Error()})
				return
			}
		case "sso":
			// SSO 验证
			if err := validateSSOLogin(req.Username, req.Password); err != nil {
				logger.Warnf("SSO login failed for username: %s, error: %v", req.Username, err)
				c.JSON(http.StatusUnauthorized, gin.H{"status": "error", "message": "SSO 登录失败"})
				return
			}
		default:
			logger.Warnf("Invalid login method: %s", user.LoginMethod)
			c.JSON(http.StatusUnauthorized, gin.H{"status": "error", "message": "无效的登录方式"})
			return
		}
	}

	// 登录成功
	currentUser := User{
		Username: user.Username,
		Role:     user.Role,
	}

	// 将用户信息存储在上下文中
	c.Set("username", user.Username)
	c.Set("current_user", currentUser)
	// 设置 Cookie
	//c.SetCookie("username", user.Username, 3600, "/", "localhost", false, true)
	logger.Infof("Login successful for username: %s", user.Username)
	loggedInUsers[user.Username] = true

	// 2. 生成JWT
	token, err := GenerateToken(user.Username, user.Role)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "生成Token失败"})
		return
	}
	setCookie(c, "username", user.Username, 7200, false, true)
	setAuthCookie(c, token)
	// 返回成功响应
	c.JSON(http.StatusOK, gin.H{
		"status":  "success",
		"message": "登录成功",
		"role":    user.Role,
		"user":    user.Username,
	})
}

type User struct {
	Username string `json:"username"`
	Password string `json:"password"`
	Role     string `json:"role"`
}

func getCurrentUser(c *gin.Context) (*User, bool) {
	// 检查用户是否登录
	currentUser, exists := c.Get("current_user")
	if !exists {
		logger.Warn("Current user not found in context")
		c.JSON(http.StatusUnauthorized, gin.H{"status": "error", "message": "用户未登录或无效！"})
		return nil, false
	}

	// 类型断言
	user, ok := currentUser.(User)
	if !ok {
		logger.Warn("Failed to assert current user type")
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "内部服务器错误"})
		return nil, false
	}

	logger.Infof("Received request from user: %s", user.Username)
	return &user, true
}

func setAuthCookie(c *gin.Context, token string) {
	domain := c.Request.Host
	if strings.Contains(domain, ":") {
		domain = strings.Split(domain, ":")[0]
	}

	// 如果是IP或localhost，不设置Domain
	var cookieDomain string
	if !isIPAddress(domain) && domain != "localhost" {
		cookieDomain = domain
	}

	c.SetCookie(
		"auth_token",
		token,
		86400, // 24小时
		"/",
		cookieDomain,
		false, // 开发环境为false，生产环境应为true（HTTPS）
		true,  // HttpOnly
	)
}

func isIPAddress(s string) bool {
	return net.ParseIP(s) != nil
}

func logout(c *gin.Context) {
	log.Println("Entered logout handler") // 调试日志

	// 1. 获取用户名（兼容旧版前端）
	var req struct {
		Username string `json:"username"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		log.Printf("Failed to bind JSON: %v\n", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}
	log.Printf("req.Username: %s\n", req.Username)

	// 2. 清除服务端状态
	delete(loggedInUsers, req.Username)

	// 3. 清除所有认证相关的Cookie
	// 清除旧版username Cookie
	c.SetCookie("username", "", -1, "/", "", false, true)

	// 清除新版auth_token Cookie（如果使用JWT）
	c.SetCookie("auth_token", "", -1, "/", "", false, true)

	// 如果是生产环境，需要设置域名（示例）：
	// c.SetCookie("auth_token", "", -1, "/", ".yourdomain.com", true, true)

	// 4. 返回响应
	log.Printf("User logged out: %s\n", req.Username)
	c.JSON(http.StatusOK, gin.H{
		"message": "Logged out successfully",
		"status":  "success",
	})
}

func checkLogin(c *gin.Context) {
	// 直接从中间件设置的上下文中获取用户
	currentUser, exists := c.Get("current_user")
	if !exists {
		c.JSON(http.StatusOK, gin.H{"logged_in": false})
		return
	}

	// 类型断言（兼容User结构体和gin.H）
	var username string
	switch user := currentUser.(type) {
	case User:
		username = user.Username
	case gin.H:
		username = user["username"].(string)
	default:
		c.JSON(http.StatusInternalServerError, gin.H{"error": "类型错误"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"logged_in": true,
		"username":  username,
	})
}

func authMiddleware(c *gin.Context) {
	logger.Info("Entered authMiddleware")

	// 从 Cookie 中获取用户名
	username, err := c.Cookie("username")
	if err != nil {
		logger.Warn("Failed to get username from cookie")
		c.JSON(http.StatusUnauthorized, gin.H{"status": "error", "message": "用户未登录或无效！"})
		c.Abort()
		return
	}

	// 模拟从数据库或缓存中获取用户信息
	user := User{
		Username: username,
		Role:     "user", // 假设角色为普通用户
	}

	// 将用户信息存储在上下文中
	c.Set("current_user", user)

	// // 1. 从Cookie获取Token
	// tokenString, err := c.Cookie("auth_token")
	// if err != nil {
	// 	c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "请先登录"})
	// 	return
	// }

	// // 2. 验证Token
	// claims, err := ParseToken(tokenString)
	// if err != nil {
	// 	c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "无效Token"})
	// 	return
	// }

	// // 3. 用户信息存入上下文
	// c.Set("current_user", gin.H{
	// 	"username": claims.Username,
	// 	"role":     claims.Role,
	// })

	c.Next()
}

// 获取所有布局配置
func getLayouts(c *gin.Context) {
	var layouts []DfLayoutOption
	if err := db.Find(&layouts).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, layouts)
}

// 创建或更新布局配置
func saveLayout(c *gin.Context) {
	var layout DfLayoutOption
	if err := c.ShouldBindJSON(&layout); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "无效的请求数据: " + err.Error()})
		return
	}

	// 验证JSON格式
	if !json.Valid(layout.LayoutJSON) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "LayoutJSON 格式无效"})
		return
	}
	if !json.Valid(layout.OptionsJSON) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "OptionsJSON 格式无效"})
		return
	}

	layout.UpdatedAt = time.Now()

	// 使用事务确保数据一致性
	err := db.Transaction(func(tx *gorm.DB) error {
		var existing DfLayoutOption

		// 查询现有记录
		query := tx.Where("layout_name = ?", layout.LayoutName).First(&existing)
		if query.Error != nil {
			if errors.Is(query.Error, gorm.ErrRecordNotFound) {
				// 新建记录
				log.Printf("创建新布局: %s", layout.LayoutName)
				return tx.Create(&layout).Error
			}
			// 查询出错
			log.Printf("查询布局失败: %v", query.Error)
			return query.Error
		}

		// 打印现有记录信息用于调试
		log.Printf("找到现有布局: %s, UpdatedAt: %d", existing.LayoutName, existing.UpdatedAt)

		// 更新现有记录
		updateData := map[string]interface{}{
			"layout_json":  layout.LayoutJSON,
			"options_json": layout.OptionsJSON,
			"updated_at":   layout.UpdatedAt,
		}

		// 使用原生SQL执行更新，便于查看实际执行的语句
		result := tx.Model(&DfLayoutOption{}).
			Where("layout_name = ?", existing.LayoutName).
			Updates(updateData)

		if result.Error != nil {
			log.Printf("更新布局失败: %v", result.Error)
			return result.Error
		}

		if result.RowsAffected == 0 {
			log.Printf("未更新任何记录，布局名: %s", existing.LayoutName)
			return fmt.Errorf("未更新任何记录")
		}

		log.Printf("成功更新布局: %s", existing.LayoutName)
		return nil
	})

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "保存失败: " + err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message": "布局保存成功",
		"data":    gin.H{"layout_name": layout.LayoutName},
	})
}

// 删除布局配置
func deleteLayout(c *gin.Context) {
	layoutName := c.Param("name")
	if err := db.Where("layout_name = ?", layoutName).Delete(&DfLayoutOption{}).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"message": "Layout deleted successfully"})
}

// // BI analysis
// type DataSource struct {
//     ID       string `json:"id"`
//     Name     string `json:"name"`
//     Type     string `json:"type"` // file, mysql, postgres, etc.
//     Path     string `json:"path"` // 文件路径或数据库连接字符串
// }
//
// type FileInfo struct {
//     Name     string `json:"name"`
//     Size     int64  `json:"size"`
//     Modified string `json:"modified"`
// }
//
// type AnalysisRequest struct {
//     DataSourceID string   `json:"dataSourceId,omitempty"`
//     FilePath     string   `json:"filePath,omitempty"`
//     FileContent  []byte   `json:"fileContent,omitempty"` // 用于上传文件
//     FileType     string   `json:"fileType"` // csv, excel, json
//     Columns      []string `json:"columns"`
//     AnalysisType string   `json:"analysisType"` // overview, correlation, etc.
// }
//
// // 获取数据目录下的文件列表
// func listDataFiles() ([]FileInfo, error) {
//     files, err := os.ReadDir("data")
//     if err != nil {
//         return nil, err
//     }
//
//     var fileInfos []FileInfo
//     for _, file := range files {
//         info, err := file.Info()
//         if err != nil {
//             continue
//         }
//
//         fileInfos = append(fileInfos, FileInfo{
//             Name:     file.Name(),
//             Size:     info.Size(),
//             Modified: info.ModTime().Format(time.RFC3339),
//         })
//     }
//
//     return fileInfos, nil
// }
//
// // 从文件加载数据
// func loadDataFromFile(filePath string, fileType string) (dataframe.DataFrame, error) {
//     file, err := os.Open(filePath)
//     if err != nil {
//         return dataframe.DataFrame{}, err
//     }
//     defer file.Close()
//
//     switch fileType {
//     case "csv":
//         return dataframe.ReadCSV(file)
//     case "excel":
//         return dataframe.ReadExcel(file)
//     case "json":
//         return dataframe.ReadJSON(file)
//     default:
//         return dataframe.DataFrame{}, fmt.Errorf("unsupported file type: %s", fileType)
//     }
// }
//
// // 处理上传的文件
// func handleUploadedFile(content []byte, fileType string) (dataframe.DataFrame, error) {
//     reader := bytes.NewReader(content)
//
//     switch fileType {
//     case "csv":
//         return dataframe.ReadCSV(reader)
//     case "excel":
//         return dataframe.ReadExcel(reader)
//     case "json":
//         return dataframe.ReadJSON(reader)
//     default:
//         return dataframe.DataFrame{}, fmt.Errorf("unsupported file type: %s", fileType)
//     }
// }
//
//
// func listFilesHandler(c *gin.Context) {
//     files, err := listDataFiles()
//     if err != nil {
//         c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
//         return
//     }
//     c.JSON(http.StatusOK, files)
// }
//
// func uploadFileHandler(c *gin.Context) {
//     file, err := c.FormFile("file")
//     if err != nil {
//         c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
//         return
//     }
//
//     // 保存文件到数据目录
//     dst := filepath.Join("data", filepath.Base(file.Filename))
//     if err := c.SaveUploadedFile(file, dst); err != nil {
//         c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
//         return
//     }
//
//     c.JSON(http.StatusOK, gin.H{"message": "File uploaded successfully", "path": dst})
// }
//
// func handleAnalysis(c *gin.Context) {
//     var req AnalysisRequest
//     if err := c.ShouldBind(&req); err != nil {
//         c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
//         return
//     }
//
//     var df dataframe.DataFrame
//     var err error
//
//     // 处理不同的数据来源
//     if req.FileContent != nil {
//         // 处理上传的文件内容
//         df, err = handleUploadedFile(req.FileContent, req.FileType)
//     } else if req.FilePath != "" {
//         // 处理指定的文件路径
//         df, err = loadDataFromFile(req.FilePath, req.FileType)
//     } else if req.DataSourceID != "" {
//         // 处理数据库源 (原有逻辑)
//         df, err = getDataFromSource(req.DataSourceID, req.TableName, req.Columns)
//     } else {
//         c.JSON(http.StatusBadRequest, gin.H{"error": "no data source specified"})
//         return
//     }
//
//     if err != nil {
//         c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
//         return
//     }
//
//     // 如果指定了列，筛选数据
//     if len(req.Columns) > 0 {
//         df = df.Select(req.Columns)
//     }
//
//     result := AnalysisResult{}
//
//     switch req.AnalysisType {
//     case "overview":
//         result.Overview = GenerateDataOverview(df)
//     case "correlation":
//         result.Correlation = CalculateCorrelation(df)
//     case "full":
//         result.Overview = GenerateDataOverview(df)
//         result.Correlation = CalculateCorrelation(df)
//     default:
//         c.JSON(http.StatusBadRequest, gin.H{"error": "invalid analysis type"})
//         return
//     }
//
//     c.JSON(http.StatusOK, result)
// }

func updateConfigIfNeeded(ip string, port string, buildDir string) {
	configPath := filepath.Join(buildDir, "config.json")

	// 检查build目录是否存在
	if _, err := os.Stat(buildDir); os.IsNotExist(err) {
		log.Println("Build directory not found, skipping config update")
		return
	}

	// 构造当前正确的API地址
	currentApiUrl := fmt.Sprintf("http://%s:%s", ip, port)

	// 检查config.json是否存在
	if _, err := os.Stat(configPath); err == nil {
		// 读取现有配置
		configData, err := ioutil.ReadFile(configPath)
		if err != nil {
			log.Printf("Failed to read config.json: %v", err)
			return
		}

		// 使用map来解析JSON，而不是固定结构体，这样可以保留所有字段
		var config map[string]interface{}
		if err := json.Unmarshal(configData, &config); err != nil {
			//log.Printf("‼️ Failed to parse config.json: %v, %s", err)
			log.Panicf("[PANIC] Failed to parse config.json: %v  , json data:%s", err, configData)
			return
		}

		// 检查apiBaseUrl是否存在且是否需要更新
		if existingUrl, ok := config["apiBaseUrl"]; ok {
			if existingUrl == currentApiUrl {
				log.Println("Config.json is up to date")
				return
			}
			log.Printf("Config.json needs update (current: %s, correct: %s)", existingUrl, currentApiUrl)
		}

		// 只更新apiBaseUrl字段
		config["apiBaseUrl"] = currentApiUrl

		// 确保build目录存在
		if err := os.MkdirAll(buildDir, 0755); err != nil {
			log.Printf("Failed to create build directory: %v", err)
			return
		}

		// 写入文件
		configData, err = json.MarshalIndent(config, "", "  ")
		if err != nil {
			log.Printf("Failed to marshal config: %v", err)
			return
		}

		if err := ioutil.WriteFile(configPath, configData, 0644); err != nil {
			log.Printf("Failed to write config.json: %v", err)
			return
		}

		log.Printf("Updated config.json with API base URL: %s", currentApiUrl)
	} else {
		// 如果文件不存在，创建新文件
		config := map[string]interface{}{
			"apiBaseUrl": currentApiUrl,
		}

		// 确保build目录存在
		if err := os.MkdirAll(buildDir, 0755); err != nil {
			log.Printf("Failed to create build directory: %v", err)
			return
		}

		// 写入文件
		configData, err := json.MarshalIndent(config, "", "  ")
		if err != nil {
			log.Printf("Failed to marshal config: %v", err)
			return
		}

		if err := ioutil.WriteFile(configPath, configData, 0644); err != nil {
			log.Printf("Failed to write config.json: %v", err)
			return
		}

		log.Printf("Created config.json with API base URL: %s", currentApiUrl)
	}
}

func getLocalIP() (string, error) {
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return "", err
	}

	for _, addr := range addrs {
		if ipnet, ok := addr.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
			// 只处理IPv4地址
			if ipnet.IP.To4() != nil {
				ip := ipnet.IP.String()

				// 排除APIPA地址 (169.254.x.x)
				if ipnet.IP.IsLinkLocalUnicast() {
					continue
				}

				// 排除虚拟机网络（可选）
				if strings.HasPrefix(ip, "192.168.98.") ||
					strings.HasPrefix(ip, "192.168.182.") {
					continue
				}

				// 优先选择无线网络适配器的地址
				if strings.HasPrefix(ip, "192.168.1.") {
					return ip, nil
				}

				return ip, nil
			}
		}
	}

	return "", fmt.Errorf("no valid local IP found")
}

//analysis
// type AnalysisRequest struct {
// 	ConfigName string `json:"config_name"`
// }

type AnalysisResponse struct {
	ConfigName        string                   `json:"config_name,omitempty"`
	FileName          string                   `json:"file_name,omitempty"`
	Shape             ShapeInfo                `json:"shape"`
	DescriptiveStats  []Stat                   `json:"descriptive_stats"`
	MissingValues     MissingInfo              `json:"missing_values"`
	NumericColumns    []string                 `json:"numeric_columns"`
	CorrelationMatrix []CorrRow                `json:"correlation_matrix"`
	PreviewData       []map[string]interface{} `json:"preview_data"`
	FeatureImportance []FeatureImportance      `json:"feature_importance,omitempty"`
	PCA               *PCAResult               `json:"pca,omitempty"`
}

type ShapeInfo struct {
	Rows    int `json:"rows"`
	Columns int `json:"columns"`
}

type Stat struct {
	Column   string  `json:"column"`
	Mean     float64 `json:"mean"`
	Std      float64 `json:"std"`
	Min      float64 `json:"min"`
	Max      float64 `json:"max"`
	Median   float64 `json:"median"`
	Unique   int     `json:"unique"`
	Missing  int     `json:"missing"`
	DataType string  `json:"data_type"`
}

type MissingInfo struct {
	Total    int             `json:"total"`
	ByColumn []MissingColumn `json:"by_column"`
}

type MissingColumn struct {
	Column  string  `json:"column"`
	Missing int     `json:"missing"`
	Percent float64 `json:"percent"`
}

type CorrRow struct {
	Column string             `json:"column"`
	Values map[string]float64 `json:"values"`
}

func handleConfigAnalysis(c *gin.Context) {
	configName := c.Param("configname")

	// In a real application, you would load the config and data from a database or file system
	// Here we'll simulate loading a CSV file based on the config name
	filePath := fmt.Sprintf("./data/%s.csv", configName)

	file, err := os.Open(filePath)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("Config not found: %s", configName)})
		return
	}
	defer file.Close()

	// Parse the CSV file
	reader := csv.NewReader(file)
	headers, err := reader.Read()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read CSV headers"})
		return
	}

	// Read all records
	var records [][]string
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read CSV record"})
			return
		}
		records = append(records, record)
	}

	// Generate analysis
	response := analyzeData(headers, records)
	response.ConfigName = configName
	response.FileName = filepath.Base(filePath)

	// Add preview data
	previewCount := 5
	if len(records) < previewCount {
		previewCount = len(records)
	}
	for i := 0; i < previewCount; i++ {
		row := make(map[string]interface{})
		for j, header := range headers {
			row[header] = records[i][j]
		}
		response.PreviewData = append(response.PreviewData, row)
	}

	c.JSON(http.StatusOK, response)
}

func handleUrlAnalysis(c *gin.Context) {
	var req struct {
		URL string `json:"url" binding:"required"`
	}

	// 解析请求体中的URL
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body: " + err.Error()})
		return
	}

	// 创建data目录（如果不存在）
	if err := os.MkdirAll("data", 0755); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create data directory"})
		return
	}

	// 从URL获取文件名
	fileName := filepath.Base(req.URL)
	filePath := filepath.Join("data", fileName)

	// 检查文件是否已存在
	if _, err := os.Stat(filePath); err == nil {
		// 文件已存在，直接使用
		fmt.Printf("File %s already exists, using existing file\n", fileName)
	} else {
		// 下载文件
		resp, err := http.Get(req.URL)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Failed to download file: " + err.Error()})
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("Failed to download file, status code: %d", resp.StatusCode)})
			return
		}

		// 创建本地文件
		outFile, err := os.Create(filePath)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create file: " + err.Error()})
			return
		}
		defer outFile.Close()

		// 保存文件内容
		_, err = io.Copy(outFile, resp.Body)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save file: " + err.Error()})
			return
		}
	}

	// 处理文件（复用上传文件的处理逻辑）
	processFile(c, filePath, fileName)
}

// 新增：抽取共用的文件处理逻辑
func processFile(c *gin.Context, filePath, fileName string) {
	// 检查文件扩展名
	ext := strings.ToLower(filepath.Ext(fileName))
	if ext != ".csv" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Only CSV files are supported"})
		return
	}

	// 打开文件
	file, err := os.Open(filePath)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to open file: " + err.Error()})
		return
	}
	defer file.Close()

	// 解析CSV
	reader := csv.NewReader(file)
	headers, err := reader.Read()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read CSV headers: " + err.Error()})
		return
	}

	// 读取所有记录
	var records [][]string
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read CSV record: " + err.Error()})
			return
		}
		records = append(records, record)
	}

	// 生成分析结果
	response := analyzeData(headers, records)
	response.FileName = fileName

	// 添加预览数据
	previewCount := 10
	if len(records) < previewCount {
		previewCount = len(records)
	}
	for i := 0; i < previewCount; i++ {
		row := make(map[string]interface{})
		for j, header := range headers {
			row[header] = records[i][j]
		}
		response.PreviewData = append(response.PreviewData, row)
	}

	c.JSON(http.StatusOK, response)
}

// 修改原有上传处理函数，复用processFile
func handleFileUpload(c *gin.Context) {
	// 获取上传的文件
	file, err := c.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "No file uploaded: " + err.Error()})
		return
	}

	// 创建data目录
	if err := os.MkdirAll("data", 0755); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create data directory"})
		return
	}

	// 保存文件
	fileName := filepath.Base(file.Filename)
	filePath := filepath.Join("data", fileName)
	if err := c.SaveUploadedFile(file, filePath); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save file: " + err.Error()})
		return
	}

	// 处理文件（复用共用逻辑）
	processFile(c, filePath, fileName)
}

// func handleFileUpload(c *gin.Context) {
// 	// Get the file from form data
// 	file, err := c.FormFile("file")
// 	if err != nil {
// 		c.JSON(http.StatusBadRequest, gin.H{"error": "No file uploaded"})
// 		return
// 	}

// 	// 创建data目录（如果不存在）
// 	if err := os.MkdirAll("data", 0755); err != nil {
// 		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create data directory"})
// 		return
// 	}

// 	// 保存文件
// 	filePath := filepath.Join("data", filepath.Base(file.Filename))
// 	if err := c.SaveUploadedFile(file, filePath); err != nil {
// 		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save file"})
// 		return
// 	}

// 	// Check file extension
// 	ext := strings.ToLower(filepath.Ext(file.Filename))
// 	if ext != ".csv" {
// 		c.JSON(http.StatusBadRequest, gin.H{"error": "Only CSV files are supported"})
// 		return
// 	}

// 	// Open the uploaded file
// 	src, err := file.Open()
// 	if err != nil {
// 		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to open uploaded file"})
// 		return
// 	}
// 	defer src.Close()

// 	// Parse the CSV file
// 	reader := csv.NewReader(src)
// 	headers, err := reader.Read()
// 	if err != nil {
// 		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read CSV headers"})
// 		return
// 	}

// 	// Read all records
// 	var records [][]string
// 	for {
// 		record, err := reader.Read()
// 		if err == io.EOF {
// 			break
// 		}
// 		if err != nil {
// 			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read CSV record"})
// 			return
// 		}
// 		records = append(records, record)
// 	}

// 	// Generate analysis
// 	response := analyzeData(headers, records)
// 	response.FileName = file.Filename

// 	// Add preview data
// 	previewCount := 10
// 	if len(records) < previewCount {
// 		previewCount = len(records)
// 	}
// 	for i := 0; i < previewCount; i++ {
// 		row := make(map[string]interface{})
// 		for j, header := range headers {
// 			row[header] = records[i][j]
// 		}
// 		response.PreviewData = append(response.PreviewData, row)
// 	}

// 	c.JSON(http.StatusOK, response)
// }

func analyzeData(headers []string, records [][]string) AnalysisResponse {
	response := AnalysisResponse{
		Shape: ShapeInfo{
			Rows:    len(records),
			Columns: len(headers),
		},
	}

	// Initialize data structures
	columnData := make(map[string][]float64)
	columnStrings := make(map[string][]string)
	isNumeric := make(map[string]bool)
	missingCount := make(map[string]int)

	// Check which columns are numeric and collect data
	for i, header := range headers {
		isNumeric[header] = true
		for _, record := range records {
			if record[i] == "" {
				missingCount[header]++
				continue
			}

			val, err := strconv.ParseFloat(record[i], 64)
			if err != nil {
				isNumeric[header] = false
				columnStrings[header] = append(columnStrings[header], record[i])
			} else {
				columnData[header] = append(columnData[header], val)
			}
		}
	}

	// Calculate descriptive statistics for numeric columns
	var numericColumns []string
	for header, numeric := range isNumeric {
		if numeric {
			numericColumns = append(numericColumns, header)
			data := columnData[header]

			stat := Stat{
				Column:   header,
				DataType: "numeric",
				Missing:  missingCount[header],
			}

			if len(data) > 0 {
				// Calculate basic stats
				sum := 0.0
				minVal := math.MaxFloat64
				maxVal := -math.MaxFloat64
				for _, val := range data {
					sum += val
					if val < minVal {
						minVal = val
					}
					if val > maxVal {
						maxVal = val
					}
				}

				stat.Mean = sum / float64(len(data))
				stat.Min = minVal
				stat.Max = maxVal

				// Calculate standard deviation
				variance := 0.0
				for _, val := range data {
					diff := val - stat.Mean
					variance += diff * diff
				}
				variance /= float64(len(data))
				stat.Std = math.Sqrt(variance)

				// Simple median calculation
				stat.Median = stat.Mean // In a real implementation, sort and find median
			}

			response.DescriptiveStats = append(response.DescriptiveStats, stat)
		} else {
			// For string columns
			uniqueValues := make(map[string]bool)
			for _, val := range columnStrings[header] {
				uniqueValues[val] = true
			}

			stat := Stat{
				Column:   header,
				DataType: "string",
				Missing:  missingCount[header],
				Unique:   len(uniqueValues),
			}
			response.DescriptiveStats = append(response.DescriptiveStats, stat)
		}
	}

	// Calculate missing values info
	totalMissing := 0
	var missingByColumn []MissingColumn
	for header, count := range missingCount {
		totalMissing += count
		percent := float64(count) / float64(len(records)) * 100
		missingByColumn = append(missingByColumn, MissingColumn{
			Column:  header,
			Missing: count,
			Percent: math.Round(percent*100) / 100,
		})
	}

	response.MissingValues = MissingInfo{
		Total:    totalMissing,
		ByColumn: missingByColumn,
	}

	response.NumericColumns = numericColumns

	// Calculate correlation matrix for numeric columns
	var corrMatrix []CorrRow
	for _, col1 := range numericColumns {
		data1 := columnData[col1]
		corrRow := CorrRow{
			Column: col1,
			Values: make(map[string]float64),
		}

		for _, col2 := range numericColumns {
			data2 := columnData[col2]
			if col1 == col2 {
				corrRow.Values[col2] = 1.0
				continue
			}

			// Pearson correlation calculation
			n := len(data1)
			if n != len(data2) || n == 0 {
				corrRow.Values[col2] = 0
				continue
			}

			var sum1, sum2, sum1Sq, sum2Sq, pSum float64
			for i := 0; i < n; i++ {
				sum1 += data1[i]
				sum2 += data2[i]
				sum1Sq += data1[i] * data1[i]
				sum2Sq += data2[i] * data2[i]
				pSum += data1[i] * data2[i]
			}

			// Calculate covariance and standard deviations
			cov := pSum - (sum1 * sum2 / float64(n))
			std1 := math.Sqrt((sum1Sq - (sum1*sum1)/float64(n)))
			std2 := math.Sqrt((sum2Sq - (sum2*sum2)/float64(n)))

			var corr float64
			if std1 != 0 && std2 != 0 {
				corr = cov / (std1 * std2)
			}

			// Ensure correlation is within [-1, 1] due to floating point precision
			corr = math.Max(-1.0, math.Min(1.0, corr))
			corrRow.Values[col2] = math.Round(corr*10000) / 10000
		}
		corrMatrix = append(corrMatrix, corrRow)
	}

	response.CorrelationMatrix = corrMatrix
	return response
}

func respondWithError(w http.ResponseWriter, code int, message string) {
	respondWithJSON(w, code, map[string]string{"error": message})
}

func respondWithJSON(w http.ResponseWriter, code int, payload interface{}) {
	response, _ := json.Marshal(payload)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	w.Write(response)
}

type AnalysisRequest struct {
	FileName string   `json:"file_name"`
	Target   string   `json:"target,omitempty"` // 目标列 (y)
	Features []string `json:"features"`         // 特征列 (X)
	Ignore   []string `json:"ignore,omitempty"` // 忽略的列
}

// 高级分析请求结构体
type AdvancedAnalysisRequest struct {
	FileName      string   `json:"file_name"`
	AnalysisTypes []string `json:"analysis_types"`
	Target        string   `json:"target,omitempty"`     // 目标列 (y)
	Features      []string `json:"features"`             // 特征列 (X)
	Ignore        []string `json:"ignore,omitempty"`     // 忽略的列
	Algorithms    []string `json:"algorithms,omitempty"` // 新增：前端传递的算法列表
}

type FeatureImportance struct {
	Feature    string  `json:"feature"`
	Importance float64 `json:"importance"`
	Type       string  `json:"type"` // tree, linear, rf, gbm
}

type PCAResult struct {
	ExplainedVariance []float64      `json:"explained_variance"`
	Components        []PCAComponent `json:"components"`
}

type PCAComponent struct {
	Feature string             `json:"feature"`
	Weights map[string]float64 `json:"weights"` // {"pc1": 0.7, "pc2": -0.2}
}

func handleAdvancedAnalysis(c *gin.Context) {
	var req AdvancedAnalysisRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// 验证配置
	if req.Target != "" && contains(req.Features, req.Target) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Target cannot be in features"})
		return
	}

	// 确保文件名有效
	if req.FileName == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Filename not provided"})
		return
	}

	// 安全地构建文件路径
	filePath := filepath.Join("data", filepath.Base(req.FileName))
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		c.JSON(http.StatusNotFound, gin.H{"error": "File not found: " + filePath})
		return
	}

	file, err := os.Open(filePath)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "File not found: " + err.Error()})
		return
	}
	defer file.Close()

	reader := csv.NewReader(file)
	headers, err := reader.Read()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read CSV headers: " + err.Error()})
		return
	}

	var records [][]string
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read CSV record: " + err.Error()})
			return
		}
		records = append(records, record)
	}

	// 添加调试信息
	log.Printf("Processing advanced analysis: %+v", req)
	log.Printf("Found %d headers and %d records", len(headers), len(records))

	response := AnalysisResponse{
		FileName: req.FileName,
		// Always include the basic analysis results
		Shape:             analyzeData(headers, records).Shape,
		DescriptiveStats:  analyzeData(headers, records).DescriptiveStats,
		MissingValues:     analyzeData(headers, records).MissingValues,
		NumericColumns:    analyzeData(headers, records).NumericColumns,
		CorrelationMatrix: analyzeData(headers, records).CorrelationMatrix,
		PreviewData:       analyzeData(headers, records).PreviewData,
	}

	for _, analysisType := range req.AnalysisTypes {
		switch analysisType {
		case "feature_importance":
			response.FeatureImportance = calculateFeatureImportance(headers, records, req.Target, req.Features, req.Algorithms)
		case "pca":
			response.PCA = performPCA(headers, records)
		}
	}

	c.JSON(http.StatusOK, response)
}

// PCA分析 (模拟实现)
func performPCA(headers []string, records [][]string) *PCAResult {
	// 转换为数值矩阵 (标准化数据)
	rows := len(records)
	cols := len(headers)
	data := mat.NewDense(rows, cols, nil)

	// 填充数据并标准化
	means := make([]float64, cols)
	stddevs := make([]float64, cols)

	for j := 0; j < cols; j++ {
		col := make([]float64, rows)
		for i := range records {
			val, _ := strconv.ParseFloat(records[i][j], 64)
			col[i] = val
		}

		mean := stat.Mean(col, nil)
		std := stat.StdDev(col, nil)
		means[j] = mean
		stddevs[j] = std

		for i := range col {
			if std != 0 {
				col[i] = (col[i] - mean) / std
			}
			data.Set(i, j, col[i])
		}
	}

	// 计算协方差矩阵
	var cov mat.SymDense
	stat.CovarianceMatrix(&cov, data, nil)

	// SVD分解 - 使用正确的常量
	var svd mat.SVD
	if !svd.Factorize(&cov, mat.SVDThin) {
		return nil
	}

	// 获取主成分
	var v mat.Dense
	svd.VTo(&v)

	// 解释方差
	s := svd.Values(nil)
	total := floats.Sum(s)
	explainedVariance := make([]float64, len(s))
	for i := range s {
		explainedVariance[i] = s[i] / total
	}

	// 提取前3个主成分
	components := make([]PCAComponent, 0)
	for j := 0; j < cols && j < 3; j++ {
		weights := make(map[string]float64)
		for i := 0; i < cols; i++ {
			weights[headers[i]] = v.At(i, j)
		}

		components = append(components, PCAComponent{
			Feature: headers[j],
			Weights: weights,
		})
	}

	return &PCAResult{
		ExplainedVariance: explainedVariance[:3], // 取前3个
		Components:        components,
	}
}

// 辅助函数：检查切片是否包含指定元素
func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

// 数据类型常量
const (
	TaskRegression     = "regression"
	TaskClassification = "classification"
)

// 识别任务类型（根据目标列数据）
func detectTaskType(targetValues []float64) string {
	// 统计唯一值数量
	unique := make(map[float64]bool)
	for _, v := range targetValues {
		if !math.IsNaN(v) {
			unique[v] = true
		}
	}
	// 唯一值少且为整数 → 分类任务
	if len(unique) <= 10 {
		for v := range unique {
			if v != math.Trunc(v) {
				return TaskRegression
			}
		}
		return TaskClassification
	}
	return TaskRegression
}

// 转换CSV数据为数值矩阵（处理缺失值）
func convertToMatrix(headers []string, records [][]string, features []int, targetIdx int) (X *mat.Dense, y mat.Matrix, taskType string) {
	nSamples := len(records)
	nFeatures := len(features)

	// 初始化矩阵（使用具体类型*mat.Dense而非接口mat.Matrix）
	X = mat.NewDense(nSamples, nFeatures, nil)
	yVals := make([]float64, nSamples)

	// 填充特征矩阵和目标向量
	for i, record := range records {
		for j, featIdx := range features {
			val, err := strconv.ParseFloat(record[featIdx], 64)
			if err != nil || math.IsNaN(val) {
				val = 0 // 简单缺失值处理，实际应用可替换为均值填充
			}
			X.Set(i, j, val)
		}

		// 处理目标值
		targetVal, err := strconv.ParseFloat(record[targetIdx], 64)
		if err != nil || math.IsNaN(targetVal) {
			targetVal = 0 // 处理目标值缺失
		}
		yVals[i] = targetVal
	}

	// 识别任务类型
	taskType = detectTaskType(yVals)
	y = mat.NewVecDense(nSamples, yVals)
	return X, y, taskType
}

// 随机森林特征重要性（分类）
// func randomForestClassificationImportance(X, y mat.Matrix, featureNames []string) []FeatureImportance {
// 	// 将矩阵转换为golearn所需的实例格式
// 	instances := matrixToInstances(X, y, featureNames, true)
// 	if instances == nil {
// 		return nil
// 	}
//
// 	// 创建随机森林分类器
// 	rf := ensemble.NewRandomForest(100, 2) // 100棵树，每棵树使用2个特征
// 	rf.Fit(instances)
//
// 	// 获取特征重要性
// 	importances := make([]FeatureImportance, len(featureNames))
// 	for i, name := range featureNames {
// 		// 计算特征重要性（基于不纯度减少）
// 		imp := rf.GetFeatureImportance(i)
// 		importances[i] = FeatureImportance{
// 			Feature:    name,
// 			Importance: imp,
// 			Type:       "rf_classification",
// 		}
// 	}
// 	return importances
// }

// 随机森林特征重要性（回归）
// func randomForestRegressionImportance(X, y mat.Matrix, featureNames []string) []FeatureImportance {
// 	// 将矩阵转换为golearn所需的实例格式
// 	instances := matrixToInstances(X, y, featureNames, false)
// 	if instances == nil {
// 		return nil
// 	}
//
// 	// 创建随机森林回归器
// 	rf := ensemble.New.NewRandomegressor{
// 		BaseEnsemble: ensemble.BaseEnsemble{
// 			NumTrees:       100,
// 			SubsampleRatio: 1.0,
// 		},
// 		TreeGenerator: trees.NewRegressionTree(2), // 每棵树使用2个特征
// 	}
// 	rf.Fit(instances)
//
// 	// 获取特征重要性
// 	importances := make([]FeatureImportance, len(featureNames))
// 	for i, name := range featureNames {
// 		// 计算特征重要性（基于方差减少）
// 		imp := rf.GetFeatureImportance(i)
// 		importances[i] = FeatureImportance{
// 			Feature:    name,
// 			Importance: imp,
// 			Type:       "rf_regression",
// 		}
// 	}
// 	return importances
// }

// 辅助函数：将矩阵转换为golearn的Instances格式
// func matrixToInstances(X, y mat.Matrix, featureNames []string, isClassification bool) base.FixedDataGrid {
// 	rows, cols := X.Dims()
//
// 	// 创建特征属性
// 	attrs := make([]base.Attribute, cols)
// 	for i := 0; i < cols; i++ {
// 		attrs[i] = base.NewFloatAttribute(featureNames[i])
// 	}
//
// 	// 创建目标属性
// 	var targetAttr base.Attribute
// 	if isClassification {
// 		targetAttr = base.NewCategoricalAttribute("target")
// 	} else {
// 		targetAttr = base.NewFloatAttribute("target")
// 	}
// 	attrs = append(attrs, targetAttr)
//
// 	// 创建数据集
// 	instanceSpecs := base.NewDenseInstances()
// 	instanceSpecs.AddAttributes(attrs...)
//
// 	// 填充数据
// 	for i := 0; i < rows; i++ {
// 		values := make([]float64, cols+1)
// 		for j := 0; j < cols; j++ {
// 			values[j] = X.At(i, j)
// 		}
// 		values[cols] = y.At(i, 0)
//
// 		instanceSpecs.AddInstance(values)
// 	}
//
// 	// 设置目标属性
// 	instanceSpecs.SetClassIndex(cols)
// 	return instanceSpecs
// }

// 线性回归特征重要性

func mutualInformation(x, y []float64) float64 {
	// 离散化数据
	discreteX := discretize(x)
	discreteY := discretize(y)

	return calculateMI(discreteX, discreteY)
}

func discretize(x []float64) []int {
	// 简单离散化为10个bin
	min := floats.Min(x)
	max := floats.Max(x)
	if min == max {
		return make([]int, len(x))
	}

	bins := make([]int, len(x))
	for i, val := range x {
		bins[i] = int(10 * (val - min) / (max - min))
		if bins[i] >= 10 {
			bins[i] = 9
		}
	}
	return bins
}

func calculateMI(x, y []int) float64 {
	n := len(x)
	if n != len(y) || n == 0 {
		return 0
	}

	// 计算联合分布和边际分布
	joint := make(map[[2]int]int)
	px := make(map[int]int)
	py := make(map[int]int)

	for i := 0; i < n; i++ {
		key := [2]int{x[i], y[i]}
		joint[key]++
		px[x[i]]++
		py[y[i]]++
	}

	// 计算互信息
	var mi float64
	for key, count := range joint {
		pXY := float64(count) / float64(n)
		pX := float64(px[key[0]]) / float64(n)
		pY := float64(py[key[1]]) / float64(n)
		if pXY > 0 && pX > 0 && pY > 0 {
			mi += pXY * math.Log(pXY/(pX*pY))
		}
	}

	return mi / math.Log(2) // 转换为比特
}

// 互信息特征重要性
func mutualInformationImportance(X *mat.Dense, y mat.Matrix, featureNames []string, taskType string) []FeatureImportance {
	importances := make([]FeatureImportance, len(featureNames))
	yVec := mat.Col(nil, 0, y)

	// 获取矩阵的行数（样本数）- 使用Dense类型的方法
	rows, _ := X.Dims()

	for i, name := range featureNames {
		xVec := make([]float64, rows)
		for j := 0; j < rows; j++ {
			// 使用At方法安全访问元素，无需直接访问RawMatrix
			xVec[j] = X.At(j, i)
		}

		// 计算互信息（这里使用自定义实现或正确的包函数）
		mi := mutualInformation(xVec, yVec)
		importances[i] = FeatureImportance{
			Feature:    name,
			Importance: mi,
			Type:       "mutual_information",
		}
	}
	return importances
}

func calculateFeatureImportance(headers []string, records [][]string, target string, features []string, algorithms []string) []FeatureImportance {
	// 1. 定位目标列和特征列
	targetIdx := -1
	for i, h := range headers {
		if h == target {
			targetIdx = i
			break
		}
	}
	if targetIdx == -1 {
		return nil
	}

	// 2. 收集特征列索引
	var featureIndices []int
	var featureNames []string
	for _, f := range features {
		for i, h := range headers {
			if h == f && i != targetIdx {
				featureIndices = append(featureIndices, i)
				featureNames = append(featureNames, f)
				break
			}
		}
	}
	if len(featureIndices) == 0 {
		return nil
	}

	// 3. 转换为数值矩阵
	X, y, taskType := convertToMatrix(headers, records, featureIndices, targetIdx)

	// 4. 根据任务类型选择算法
	var allImportances []FeatureImportance
	for _, algo := range algorithms {
		switch algo {
		case "rf":
			if taskType == TaskClassification {
				allImportances = append(allImportances, randomForestClassificationImportance(X, y, featureNames)...)
			} else {
				allImportances = append(allImportances, randomForestRegressionImportance(X, y, featureNames)...)
			}
		case "linear":
			if taskType == TaskClassification {
				allImportances = append(allImportances, logisticRegressionImportance(X, y, featureNames)...)
			} else {
				allImportances = append(allImportances, linearRegressionImportance(X, y, featureNames)...)
			}
		case "mi":
			allImportances = append(allImportances, mutualInformationImportance(X, y, featureNames, taskType)...)
			//case "tree":
			//             // 决策树算法实现
			//             allImportances = append(allImportances, decisionTreeImportance(X, y, featureNames, taskType)...)
		}
	}

	// 5. 标准化每个算法的重要性分数
	// 按算法分组
	algorithmGroups := make(map[string][]FeatureImportance)
	for _, imp := range allImportances {
		algorithmGroups[imp.Type] = append(algorithmGroups[imp.Type], imp)
	}

	// 标准化每组分数到[0,1]
	normalized := make([]FeatureImportance, 0, len(allImportances))
	for _, group := range algorithmGroups {
		// 找到最大值
		maxVal := 0.0
		for _, imp := range group {
			if imp.Importance > maxVal {
				maxVal = imp.Importance
			}
		}
		// 标准化
		for _, imp := range group {
			normImp := imp
			if maxVal > 0 {
				normImp.Importance = imp.Importance / maxVal
			}
			normalized = append(normalized, normImp)
		}
	}

	return normalized
}

type InferFetchResult struct {
	ID            int64      `gorm:"primaryKey;autoIncrement" json:"id"`               // 唯一自增ID（时间相关，便于排序查询）
	TriggerSource string     `gorm:"size:20;not null;index" json:"triggerSource"`      // 触发来源：service/test_web
	TaskID        string     `gorm:"size:100;not null;uniqueIndex" json:"taskId"`      // 推理任务唯一标识
	TaskName      string     `gorm:"size:255;not null" json:"taskName"`                // 任务名称（对应原ConfigName）
	Type          string     `gorm:"size:50;not null;index" json:"type"`               // 类型：sql/normal
	BaseTaskId    string     `gorm:"size:100;not null;index" json:"baseTaskId"`        // 关联的基础跑批任务ID（用于溯源）
	ResultJSON    string     `gorm:"type:text" json:"resultJSON"`                      // 数值结果（JSON格式，核心数据）
	Supplement    string     `gorm:"type:text" json:"supplement"`                      // 补充说明（简短）
	StartTime     time.Time  `gorm:"not null;index" json:"startTime"`                  // 开始时间
	EndTime       *time.Time `json:"endTime,omitempty"`                                // 结束时间
	Status        string     `gorm:"size:20;default:'processing';index" json:"status"` // 状态
	ErrorMsg      string     `gorm:"type:text" json:"errorMsg,omitempty"`              // 错误信息（仅失败时存储）
	CreatedAt     time.Time  `gorm:"autoCreateTime" json:"createdAt"`                  // 记录创建时间
}

// 表名映射
func (InferFetchResult) TableName() string {
	return "infer_fetch_results"
}

// 推理取数请求结构体（精简版）
type InferFetchRequest struct {
	TaskId        string                 `json:"taskid" binding:"required"` // 关联的基础跑批任务ID
	InferParams   map[string]interface{} `json:"inferparams"`               // 推理参数
	TriggerSource string                 `json:"triggersource"`             // 触发来源
}

type InferFetchBaseTask struct {
	ID         string
	Configname string
	Type       string // 配置类型：sql/normal
	Status     string
	SqlJson    datatypes.JSON // SQL配置（仅sql类型用）
}

func inferFetchData(c *gin.Context) {
	log.Println("========== 开始处理推理实时取数请求 ==========")

	var request InferFetchRequest
	if err := c.ShouldBindJSON(&request); err != nil {
		log.Printf("❌ 推理请求解析失败: %v", err)
		c.JSON(http.StatusBadRequest, gin.H{
			"success": false,
			"error":   "无效的请求数据: " + err.Error(),
		})
		return
	}

	// 1. 查询关联的基础跑批任务（获取核心配置）
	var baseTask InferFetchBaseTask
	if err := db.Model(&DataFetchTask{}).
		Where("id = ?", request.TaskId).
		First(&baseTask).Error; err != nil {
		log.Printf("❌ 基础任务不存在: %s, 错误: %v", request.TaskId, err)
		c.JSON(http.StatusBadRequest, gin.H{
			"success": false,
			"error":   "关联的跑批任务不存在",
		})
		return
	}

	// 校验基础任务状态（必须已完成）
	if baseTask.Status != "completed" {
		log.Printf("❌ 基础任务状态异常: %s, 状态: %s", request.TaskId, baseTask.Status)
		c.JSON(http.StatusBadRequest, gin.H{
			"success": false,
			"error":   "只能基于已完成的跑批任务进行推理",
		})
		return
	}

	// 2. 创建推理任务记录（InferFetchResult）
	taskID := fmt.Sprintf("infer%s--%s", time.Now().Format("060102150405"), request.TaskId)
	inferRecord := InferFetchResult{
		TaskID:        taskID,
		TaskName:      baseTask.Configname,
		TriggerSource: request.TriggerSource,
		Type:          baseTask.Type,
		BaseTaskId:    request.TaskId,
		StartTime:     time.Now(),
		Status:        "processing",
	}
	if err := db.Create(&inferRecord).Error; err != nil {
		log.Printf("❌ 创建推理记录失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"success": false,
			"error":   "创建推理任务失败",
		})
		return
	}
	log.Printf("✅ 推理任务记录创建成功: %s（记录ID: %d）", taskID, inferRecord.ID)

	// 3. 生成推理配置文件（供处理器读取）
	// 3.1 创建配置目录
	if err := os.MkdirAll("taskconfig/infer", 0755); err != nil {
		log.Printf("❌ 创建推理配置目录失败: %v", err)
		updateInferTaskStatus(taskID, "failed", "配置目录创建失败")
		c.JSON(http.StatusInternalServerError, gin.H{
			"success": false,
			"error":   "推理配置准备失败",
		})
		return
	}

	// 3.2 构建推理配置数据（包含基础任务信息和推理参数）
	configData := map[string]interface{}{
		"baseTaskId":    request.TaskId,           // 关联的基础任务ID
		"inferRecordId": inferRecord.ID,           // 推理记录ID（用于状态更新）
		"inferParams":   request.InferParams,      // 推理参数
		"baseTaskType":  baseTask.Type,            // 基础任务类型
		"sqlJson":       string(baseTask.SqlJson), // SQL配置（仅sql类型用）
	}
	configJSON, err := json.Marshal(configData)
	if err != nil {
		log.Printf("❌ 序列化推理配置失败: %v", err)
		updateInferTaskStatus(taskID, "failed", "配置序列化失败")
		c.JSON(http.StatusInternalServerError, gin.H{
			"success": false,
			"error":   "推理配置准备失败",
		})
		return
	}

	// 3.3 写入配置文件
	configPath := fmt.Sprintf("taskconfig/infer/%s.json", taskID)
	if err := os.WriteFile(configPath, configJSON, 0644); err != nil {
		log.Printf("❌ 写入推理配置文件失败: %v", err)
		updateInferTaskStatus(taskID, "failed", "配置文件写入失败")
		c.JSON(http.StatusInternalServerError, gin.H{
			"success": false,
			"error":   "推理配置准备失败",
		})
		return
	}
	log.Printf("✅ 推理配置文件生成: %s", configPath)

	// 4. 调用统一处理器 runFetchData 启动异步处理
	// - 训练请求传空（推理模式无需）
	// - 推理请求传当前request
	// - 模式指定为"infer"
	var emptyTrainRequest DataFetchRequest // 空的训练请求
	// runFetchData(
	// 	baseTask.Type,     // 配置类型（sql/normal，与基础任务一致）
	// 	taskID,            // 推理任务ID
	// 	configPath,        // 推理配置文件路径
	// 	emptyTrainRequest, // 空训练请求
	// 	request,           // 推理请求
	// 	FetchModeInfer,    // 推理模式
	// )
	result := runFetchData(
		baseTask.Type,
		taskID,
		configPath,
		emptyTrainRequest,
		baseTask,
		FetchModeInfer,
	)

	if result.Error != nil {
		// 失败：更新任务状态，返回错误
		updateInferTaskStatus(taskID, "failed", result.Error.Error())
		c.JSON(http.StatusInternalServerError, gin.H{
			"success": false,
			"error":   result.Error.Error(),
			"taskId":  taskID,
		})
		return
	}

	// 成功：更新任务状态，返回dataframe
	updateInferTaskStatus(taskID, "completed", "数据处理完成")
	c.JSON(http.StatusOK, gin.H{
		"success":   true,
		"taskId":    taskID,
		"dataframe": result.Data, // 返回解析后的dataframe
		"message":   "推理取数完成",
	})
	log.Printf("========== 推理任务处理流程结束: %s ==========", taskID)
}

type QueryConfigRequest struct {
	TaskId string `json:"taskId" binding:"required"` // 必传参数校验
}

// 定义响应结构体
type CommonResponse struct {
	Success bool        `json:"success"`
	Message string      `json:"message,omitempty"`
	Data    interface{} `json:"data,omitempty"`
}

// 处理查询配置名的Post接口
func QueryConfigNameHandler(c *gin.Context) {
	var req QueryConfigRequest
	// 解析并校验请求参数
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, CommonResponse{
			Success: false,
			Message: "无效的请求参数：" + err.Error(),
		})
		return
	}

	// 从数据库查询任务信息
	var task DataFetchTask
	// 根据ID查询（注意：这里的ID对应前端传入的taskId）
	result := db.Where("id = ?", req.TaskId).First(&task)
	if result.Error != nil {
		// 处理查询错误（包括记录不存在的情况）
		if errors.Is(result.Error, gorm.ErrRecordNotFound) {
			c.JSON(http.StatusOK, CommonResponse{
				Success: false,
				Message: "未找到对应的任务记录",
			})
		} else {
			c.JSON(http.StatusInternalServerError, CommonResponse{
				Success: false,
				Message: "数据库查询失败：" + result.Error.Error(),
			})
		}
		return
	}

	// 返回查询到的配置名
	c.JSON(http.StatusOK, CommonResponse{
		Success: true,
		Data: map[string]string{
			"configName": task.ConfigName, // 返回ConfigName字段
			"taskId":     task.ID,         // 可额外返回任务ID供前端校验
		},
	})
}

// InferFetchResult 对应数据库表结构的结构体

// QueryInferHistoryRequest 定义查询参数结构体
// QueryInferHistoryRequest 定义查询参数结构体
type QueryInferHistoryRequest struct {
	ID            string `json:"id"`                  // 推理记录ID
	TaskID        string `json:"taskId"`              // 任务ID
	BaseTaskId    string `json:"baseTaskId"`          // 基础任务ID
	TriggerSource string `json:"triggerSource"`       // 触发来源
	Status        string `json:"status"`              // 状态
	StartTime     string `json:"startTime"`           // 开始时间
	EndTime       string `json:"endTime"`             // 结束时间
	Page          int    `json:"page,default=1"`      // 页码
	PageSize      int    `json:"pageSize,default=10"` // 每页条数
}

// QueryInferHistoryResponse 定义响应结构体
type QueryInferHistoryResponse struct {
	Success bool               `json:"success"`
	Message string             `json:"message,omitempty"`
	Data    []InferFetchResult `json:"data,omitempty"`
	Total   int64              `json:"total,omitempty"`
}

// QueryInferHistory 处理历史推理记录查询请求（POST方式）
func QueryInferHistory(c *gin.Context) {
	var req QueryInferHistoryRequest

	// 解析请求参数
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, QueryInferHistoryResponse{
			Success: false,
			Message: "无效的查询参数: " + err.Error(),
		})
		return
	}

	// 验证分页参数
	if req.Page < 1 {
		req.Page = 1
	}
	if req.PageSize < 1 || req.PageSize > 100 {
		req.PageSize = 10
	}

	// 关键修复：显式指定表名，确保GORM能正确识别
	mainQuery := db.Table((&InferFetchResult{}).TableName())

	// 构建查询条件
	if req.ID != "" {
		id, err := strconv.ParseInt(req.ID, 10, 64)
		if err == nil {
			mainQuery = mainQuery.Where("id = ?", id)
		} else {
			c.JSON(http.StatusBadRequest, QueryInferHistoryResponse{
				Success: false,
				Message: "ID必须是整数",
			})
			return
		}
	}

	if req.TaskID != "" {
		mainQuery = mainQuery.Where("task_id LIKE ?", "%"+req.TaskID+"%")
	}

	if req.BaseTaskId != "" {
		mainQuery = mainQuery.Where("base_task_id LIKE ?", "%"+req.BaseTaskId+"%")
	}

	if req.TriggerSource != "" {
		mainQuery = mainQuery.Where("trigger_source = ?", req.TriggerSource)
	}

	if req.Status != "" {
		mainQuery = mainQuery.Where("status = ?", req.Status)
	}

	// 处理时间范围查询
	if req.StartTime != "" {
		startTime, err := time.Parse("2006-01-02 15:04:05", req.StartTime)
		if err != nil {
			c.JSON(http.StatusBadRequest, QueryInferHistoryResponse{
				Success: false,
				Message: "开始时间格式错误，应为YYYY-MM-DD HH:mm:ss",
			})
			return
		}
		mainQuery = mainQuery.Where("start_time >= ?", startTime)
	}

	if req.EndTime != "" {
		endTime, err := time.Parse("2006-01-02 15:04:05", req.EndTime)
		if err != nil {
			c.JSON(http.StatusBadRequest, QueryInferHistoryResponse{
				Success: false,
				Message: "结束时间格式错误，应为YYYY-MM-DD HH:mm:ss",
			})
			return
		}
		mainQuery = mainQuery.Where("start_time <= ?", endTime)
	}

	// 关键修复：使用明确的表名查询总数
	var total int64
	if err := mainQuery.Count(&total).Error; err != nil {
		c.JSON(http.StatusInternalServerError, QueryInferHistoryResponse{
			Success: false,
			Message: "查询总数失败: " + err.Error(),
		})
		return
	}

	// 计算分页偏移量
	offset := (req.Page - 1) * req.PageSize

	// 执行分页查询，按ID降序排序
	var results []InferFetchResult
	if err := mainQuery.Order("id DESC").Offset(offset).Limit(req.PageSize).Find(&results).Error; err != nil {
		c.JSON(http.StatusInternalServerError, QueryInferHistoryResponse{
			Success: false,
			Message: "查询记录失败: " + err.Error(),
		})
		return
	}

	// 返回成功响应
	c.JSON(http.StatusOK, QueryInferHistoryResponse{
		Success: true,
		Data:    results,
		Total:   total,
	})
}

// 角色模型
type PDataUserRole struct {
	RoleKey     string    `gorm:"primaryKey;size:50" json:"roleKey"`
	RoleName    string    `gorm:"size:100;not null" json:"roleName"`
	Description string    `gorm:"size:500" json:"description"`
	CreatedAt   time.Time `json:"createdAt"`
	UpdatedAt   time.Time `json:"updatedAt"`
	UserCount   int       `gorm:"-" json:"userCount"` // 非数据库字段，用于前端显示
}

// 权限项模型
type PDataPermission struct {
	ID          uint   `gorm:"primaryKey" json:"id"`
	Key         string `gorm:"size:50;uniqueIndex;not null" json:"key"`
	Name        string `gorm:"size:100;not null" json:"name"`
	Description string `gorm:"size:500" json:"description"`
	Category    string `gorm:"size:50" json:"category"` // 权限类别：功能权限/数据权限等
}

// 角色权限关联模型
type PDataRolePermission struct {
	RoleKey       string    `gorm:"primaryKey;size:50" json:"roleKey"`
	PermissionKey string    `gorm:"primaryKey;size:50" json:"permissionKey"`
	Restrictions  string    `gorm:"type:text" json:"restrictions"` // 权限限制条件（JSON）
	CreatedAt     time.Time `json:"createdAt"`
}

// 角色数据权限模型（表级）
type PDataRoleDataPermission struct {
	RoleKey         string    `gorm:"primaryKey;size:50" json:"roleKey"`
	TableName       string    `gorm:"primaryKey;size:100" json:"tableName"`
	AllowRead       bool      `gorm:"default:true" json:"allowRead"`
	AllowWrite      bool      `gorm:"default:false" json:"allowWrite"`
	FilterCondition string    `gorm:"size:500" json:"filterCondition"` // 行级过滤条件
	CreatedAt       time.Time `json:"createdAt"`
}

// SQL权限限制模型
type PDataRoleSQLRestriction struct {
	RoleKey         string    `gorm:"primaryKey;size:50" json:"roleKey"`
	RestrictionType string    `gorm:"primaryKey;size:50" json:"restrictionType"` // sql_select, sql_update等
	Config          string    `gorm:"size:500" json:"config"`                    // 限制配置，如最大行数
	CreatedAt       time.Time `json:"createdAt"`
}

// 注册权限相关路由

// 获取所有角色
func getRoles(c *gin.Context) {
	var roles []PDataUserRole
	if err := db.Find(&roles).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取角色列表失败"})
		return
	}

	// 统计每个角色的用户数量（简化实现）
	for i := range roles {
		var count int64
		db.Model(&PDataUserRole{}).Where("role_key = ?", roles[i].RoleKey).Count(&count)
		roles[i].UserCount = int(count)
	}

	c.JSON(http.StatusOK, roles)
}

// 创建角色
func createRole(c *gin.Context) {
	var role PDataUserRole
	if err := c.ShouldBindJSON(&role); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "无效的角色数据"})
		return
	}

	if err := db.Create(&role).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "创建角色失败"})
		return
	}

	c.JSON(http.StatusOK, role)
}

// 获取角色权限配置
func getRolePermissions(c *gin.Context) {
	roleKey := c.Param("roleKey")

	// 获取功能权限
	var rolePerms []PDataRolePermission
	db.Where("role_key = ?", roleKey).Find(&rolePerms)
	funcPerms := make([]string, len(rolePerms))
	for i, rp := range rolePerms {
		funcPerms[i] = rp.PermissionKey
	}

	// 获取数据权限（表级）
	var dataPerms []PDataRoleDataPermission
	db.Where("role_key = ?", roleKey).Find(&dataPerms)
	tables := make([]string, len(dataPerms))
	for i, dp := range dataPerms {
		tables[i] = dp.TableName
	}

	// 获取SQL限制
	var sqlRestrictions []PDataRoleSQLRestriction
	db.Where("role_key = ?", roleKey).Find(&sqlRestrictions)
	sqlRestricts := make([]string, len(sqlRestrictions))
	for i, sr := range sqlRestrictions {
		sqlRestricts[i] = sr.RestrictionType
	}

	c.JSON(http.StatusOK, gin.H{
		"functionalPermissions": funcPerms,
		"dataPermissions":       tables,
		"sqlRestrictions":       sqlRestricts,
	})
}

// 保存角色权限配置
func saveRolePermissions(c *gin.Context) {
	roleKey := c.Param("roleKey")
	var req struct {
		FunctionalPermissions []string `json:"functionalPermissions"`
		DataPermissions       []string `json:"dataPermissions"`
		SqlRestrictions       []string `json:"sqlRestrictions"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "无效的权限配置数据"})
		return
	}

	// 开启事务
	tx := db.Begin()
	defer func() {
		if r := recover(); r != nil {
			tx.Rollback()
		}
	}()

	// 先删除原有权限配置
	tx.Where("role_key = ?", roleKey).Delete(&PDataRolePermission{})
	tx.Where("role_key = ?", roleKey).Delete(&PDataRoleDataPermission{})
	tx.Where("role_key = ?", roleKey).Delete(&PDataRoleSQLRestriction{})

	// 保存功能权限
	for _, permKey := range req.FunctionalPermissions {
		tx.Create(&PDataRolePermission{
			RoleKey:       roleKey,
			PermissionKey: permKey,
		})
	}

	// 保存数据权限（默认只允许读取）
	for _, tableName := range req.DataPermissions {
		tx.Create(&PDataRoleDataPermission{
			RoleKey:   roleKey,
			TableName: tableName,
			AllowRead: true,
		})
	}

	// 保存SQL限制
	for _, rt := range req.SqlRestrictions {
		config := ""
		if rt == "sql_row_limit" {
			config = "100" // 默认限制100行
		}
		tx.Create(&PDataRoleSQLRestriction{
			RoleKey:         roleKey,
			RestrictionType: rt,
			Config:          config,
		})
	}

	if err := tx.Commit().Error; err != nil {
		tx.Rollback()
		c.JSON(http.StatusInternalServerError, gin.H{"error": "保存权限配置失败"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"status": "success", "message": "权限配置已保存"})
}

// SQL查询权限校验中间件
func SQLPermissionMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// 获取当前用户角色
		userID := c.MustGet("userID").(string)
		var userRoles []PDataUserRole
		if err := db.Where("user_id = ?", userID).Find(&userRoles).Error; err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "获取用户角色失败"})
			c.Abort()
			return
		}

		// 获取请求中的SQL
		var req struct {
			SQL       string `json:"sql"`
			TableName string `json:"tableName"`
		}
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "无效的请求参数"})
			c.Abort()
			return
		}

		// 检查用户是否有权限执行该SQL
		hasPermission := false
		sqlType := getSQLType(req.SQL) // 自定义函数：判断SQL类型（SELECT/UPDATE/DELETE等）

		for _, ur := range userRoles {
			var restrictions []PDataRoleSQLRestriction
			db.Where("role_key = ?", ur.RoleKey).Find(&restrictions)

			// 检查是否有对应SQL类型的权限
			for _, r := range restrictions {
				if r.RestrictionType == sqlType ||
					(sqlType == "sql_select" && r.RestrictionType == "sql_row_limit") {
					hasPermission = true
					break
				}
			}
			if hasPermission {
				break
			}
		}

		if !hasPermission {
			c.JSON(http.StatusForbidden, gin.H{"error": "没有执行该SQL操作的权限"})
			c.Abort()
			return
		}

		c.Next()
	}
}

// 更新角色信息
func updateRole(c *gin.Context) {
	roleKey := c.Param("roleKey")

	var role PDataUserRole
	if err := c.ShouldBindJSON(&role); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "无效的角色数据", "details": err.Error()})
		return
	}

	// 检查角色是否存在
	var existingRole PDataUserRole
	if err := db.First(&existingRole, "role_key = ?", roleKey).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			c.JSON(http.StatusNotFound, gin.H{"error": "角色不存在"})
		} else {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "查询角色失败"})
		}
		return
	}

	// 确保角色标识不可修改
	role.RoleKey = roleKey

	// 更新角色信息
	if err := db.Save(&role).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "更新角色失败", "details": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":  "success",
		"message": "角色更新成功",
		"data":    role,
	})
}

// 删除角色
func deleteRole(c *gin.Context) {
	roleKey := c.Param("roleKey")

	// 检查角色是否存在
	var role PDataUserRole
	if err := db.First(&role, "role_key = ?", roleKey).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			c.JSON(http.StatusNotFound, gin.H{"error": "角色不存在"})
		} else {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "查询角色失败"})
		}
		return
	}

	// 检查是否有关联用户
	var userRoleCount int64
	if err := db.Model(&PDataUserRole{}).Where("role_key = ?", roleKey).Count(&userRoleCount).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "检查角色关联用户失败"})
		return
	}

	if userRoleCount > 0 {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "无法删除角色，该角色仍关联有用户",
		})
		return
	}

	// 开启事务删除角色及关联权限
	tx := db.Begin()
	defer func() {
		if r := recover(); r != nil {
			tx.Rollback()
		}
	}()

	// 删除角色关联的权限
	if err := tx.Where("role_key = ?", roleKey).Delete(&PDataRolePermission{}).Error; err != nil {
		tx.Rollback()
		c.JSON(http.StatusInternalServerError, gin.H{"error": "删除角色权限失败"})
		return
	}

	// 删除角色关联的数据权限
	if err := tx.Where("role_key = ?", roleKey).Delete(&PDataRoleDataPermission{}).Error; err != nil {
		tx.Rollback()
		c.JSON(http.StatusInternalServerError, gin.H{"error": "删除角色数据权限失败"})
		return
	}

	// 删除角色关联的SQL限制
	if err := tx.Where("role_key = ?", roleKey).Delete(&PDataRoleSQLRestriction{}).Error; err != nil {
		tx.Rollback()
		c.JSON(http.StatusInternalServerError, gin.H{"error": "删除角色SQL限制失败"})
		return
	}

	// 删除角色本身
	if err := tx.Delete(&role).Error; err != nil {
		tx.Rollback()
		c.JSON(http.StatusInternalServerError, gin.H{"error": "删除角色失败"})
		return
	}

	if err := tx.Commit().Error; err != nil {
		tx.Rollback()
		c.JSON(http.StatusInternalServerError, gin.H{"error": "提交事务失败"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":  "success",
		"message": "角色删除成功",
	})
}

// 获取所有权限项
func getPermissions(c *gin.Context) {
	var permissions []PDataPermission

	// 可选：按类别筛选
	category := c.Query("category")
	query := db.Model(&PDataPermission{})
	if category != "" {
		query = query.Where("category = ?", category)
	}

	if err := query.Find(&permissions).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取权限列表失败", "details": err.Error()})
		return
	}

	c.JSON(http.StatusOK, permissions)
}

// 判断SQL语句类型
func getSQLType(sql string) string {
	// 去除SQL语句中的注释
	cleanSQL := removeSQLComments(sql)

	// 提取SQL开头的关键字
	tokens := strings.Fields(strings.ToUpper(cleanSQL))
	if len(tokens) == 0 {
		return "unknown"
	}

	switch tokens[0] {
	case "SELECT", "SHOW", "DESCRIBE", "EXPLAIN":
		return "sql_select"
	case "INSERT", "REPLACE":
		return "sql_insert"
	case "UPDATE":
		return "sql_update"
	case "DELETE", "TRUNCATE":
		return "sql_delete"
	case "CREATE", "ALTER", "DROP", "RENAME", "COMMENT":
		return "sql_ddl"
	case "GRANT", "REVOKE":
		return "sql_grant"
	default:
		return "unknown"
	}
}

// 辅助函数：移除SQL注释
func removeSQLComments(sql string) string {
	// 简单处理单行和多行注释，实际场景可能需要更复杂的处理
	re := regexp.MustCompile(`(--.*$)|(/\*.*?\*/)`)
	return re.ReplaceAllStringFunc(sql, func(match string) string {
		return ""
	})
}

// 唯一的数据结构：操作及其关联用户
type POperationWithUsers struct {
	Opkey     string    `gorm:"primaryKey;size:100" json:"opkey"` // 替换 key 为 opKey
	Name      string    `gorm:"size:100;not null" json:"name"`
	Users     string    `gorm:"type:text" json:"users"`
	Modifier  string    `gorm:"type:text" json:"modifier"` // 修正为 modifier
	CreatedAt time.Time `json:"createdAt"`
	UpdatedAt time.Time `json:"updatedAt"`
}

// 创建操作（同时设置关联用户）
func createOperation(c *gin.Context) {
	var req struct {
		Key   string `json:"key" binding:"required"`
		Name  string `json:"name" binding:"required"`
		Users string `json:"users"` // 可选，用户ID逗号分隔
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "无效参数: " + err.Error()})
		return
	}

	// 检查是否已存在
	var existing POperationWithUsers
	if err := db.Where("opkey = ?", req.Key).First(&existing).Error; err == nil {
		c.JSON(http.StatusConflict, gin.H{"error": "操作标识已存在"})
		return
	}

	operation := POperationWithUsers{
		Opkey: req.Key,
		Name:  req.Name,
		Users: req.Users, // 直接存储逗号分隔的用户字符串
	}

	if err := db.Create(&operation).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "创建失败: " + err.Error()})
		return
	}

	c.JSON(http.StatusOK, operation)
}

// 2. 获取所有操作
func getOperations(c *gin.Context) {

	var operations []POperationWithUsers

	// 查询所有记录
	if err := db.Find(&operations).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "查询失败: " + err.Error()})
		return
	}

	c.JSON(http.StatusOK, operations)
}

// 3. 更新操作
func updateOperation(c *gin.Context) {

	key := c.Param("key") // 从URL获取操作标识
	var req struct {
		Name  string `json:"name" binding:"required"` // 操作名称（必填）
		Users string `json:"users"`                   // 用户ID（可选）
	}

	// 绑定请求体
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "参数错误: " + err.Error()})
		return
	}
	log.Printf("key:%s", key)
	// 查找操作是否存在
	var operation POperationWithUsers
	if err := db.Where(" opkey  = ?", key).First(&operation).Error; err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "操作不存在"})
		return
	}

	// 更新字段
	operation.Name = req.Name
	operation.Users = req.Users
	operation.UpdatedAt = time.Now()

	// 保存更新
	if err := db.Save(&operation).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "更新失败: " + err.Error()})
		return
	}

	c.JSON(http.StatusOK, operation)
}

// 4. 删除操作
func deleteOperation(c *gin.Context) {

	key := c.Param("key")
	log.Printf("key:%s", key)
	// 直接删除
	if err := db.Delete(&POperationWithUsers{}, "opkey = ?", key).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "删除失败: " + err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "删除成功"})
}

// 辅助函数：去除文件名的扩展名（如 "myapp.exe" -> "myapp"）
func removeExtension(filename string) string {
	ext := filepath.Ext(filename)
	if ext == "" {
		return filename
	}
	return filename[:len(filename)-len(ext)]
}

// 释放资源到目标目录
func releaseResources(targetDir string) error {
	// 创建目标目录
	if err := os.MkdirAll(targetDir, 0755); err != nil {
		return fmt.Errorf("创建目录失败：%w", err)
	}

	// 释放前端资源到 targetDir/build
	frontendFS, err := fs.Sub(resources, "web/build")
	if err != nil {
		return fmt.Errorf("提取前端资源失败：%w", err)
	}
	buildDir := filepath.Join(targetDir, "build")
	if err := releaseFS(frontendFS, ".", buildDir); err != nil {
		return fmt.Errorf("释放前端资源失败：%w", err)
	}

	// 释放配置文件到 targetDir/config
	configFS, err := fs.Sub(resources, "config")
	if err != nil {
		return fmt.Errorf("提取配置资源失败：%w", err)
	}
	configDir := filepath.Join(targetDir, "config")
	if err := releaseFS(configFS, ".", configDir); err != nil {
		return fmt.Errorf("释放配置资源失败：%w", err)
	}

	filePath := "datafactory_fetchdata.py"
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		// 文件不存在，才进行释放
		content, err := fs.ReadFile(resources, filePath)
		if err != nil {
			return fmt.Errorf("提取配置资源失败：%w", err)
		}

		if err := os.WriteFile(filePath, content, 0644); err != nil {
			return fmt.Errorf("释放脚本文件失败：%w", err)
		}
	} else if err != nil {
		// 其他错误情况
		return fmt.Errorf("检查文件是否存在失败：%w", err)
	}
    filePath2 := "genCustomReport.py"
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		// 文件不存在，才进行释放
		content, err := fs.ReadFile(resources, filePath2)
		if err != nil {
			return fmt.Errorf("提取配置资源失败：%w", err)
		}

		if err := os.WriteFile(filePath2, content, 0644); err != nil {
			return fmt.Errorf("释放脚本文件失败：%w", err)
		}
	} else if err != nil {
		// 其他错误情况
		return fmt.Errorf("检查文件是否存在失败：%w", err)
	}


	// if err := os.WriteFile("datafactory_fetchdata.py", content, 0644); err != nil {
	// 	return fmt.Errorf("释放脚本文件失败：%w", err)
	// }

	return nil
}

// 工具函数：释放嵌入的文件系统到目标目录
func releaseFS(embeddedFS fs.FS, sourcePath string, targetDir string) error {
	return fs.WalkDir(embeddedFS, sourcePath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		targetPath := filepath.Join(targetDir, path)

		// 检查目标路径是否已存在
		exists, err := pathExists(targetPath)
		if err != nil {
			return fmt.Errorf("检查路径 %s 失败：%w", targetPath, err)
		}

		if exists {
			// 路径已存在，直接跳过（可选项：打印日志便于调试）
			// fmt.Printf("路径已存在，跳过：%s\n", targetPath)
			return nil
		}

		// 路径不存在，执行创建/写入操作
		if d.IsDir() {
			return os.MkdirAll(targetPath, 0755)
		}

		data, err := fs.ReadFile(embeddedFS, path)
		if err != nil {
			return fmt.Errorf("读取文件 %s 失败：%w", path, err)
		}

		return os.WriteFile(targetPath, data, 0644)
	})
}

// 辅助函数：检查路径是否存在
func pathExists(path string) (bool, error) {
	_, err := os.Stat(path)
	if err == nil {
		// 路径存在
		return true, nil
	}
	if os.IsNotExist(err) {
		// 路径不存在
		return false, nil
	}
	// 其他错误（如权限问题）
	return false, err
}

func releaseSelf(targetDir string) error {
	// 1. 获取当前可执行文件的路径
	exePath, err := os.Executable()
	if err != nil {
		return err
	}

	// 2. 打开当前可执行文件
	srcFile, err := os.Open(exePath)
	if err != nil {
		return err
	}
	defer srcFile.Close()

	// 3. 在目标目录创建同名可执行文件
	exeName := filepath.Base(exePath)
	dstPath := filepath.Join(targetDir, exeName)
	dstFile, err := os.Create(dstPath)
	if err != nil {
		return err
	}
	defer dstFile.Close()

	// 4. 复制文件内容（二进制）
	if _, err := io.Copy(dstFile, srcFile); err != nil {
		return err
	}

	// 5. 赋予可执行权限（Linux/macOS 必需）
	if err := os.Chmod(dstPath, 0755); err != nil {
		return err
	}

	fmt.Printf("自身可执行文件已释放到：%s\n", dstPath)
	return nil
}

// ///////////
// PDataTask 任务数据模型

// PDataTask 增强版任务数据模型，添加状态详情、运行结果和扩展字段
type PDataTask struct {
	ID           string          `gorm:"primaryKey" json:"id"`
	TaskName     string          `json:"taskName"`
	TaskType     string          `json:"taskType"`     // sql, python, email
	Status       string          `json:"status"`       // pending, running, completed, failed
	StatusDetail string          `json:"statusDetail"` // 状态详情，记录错误信息等
	Result       json.RawMessage `json:"result"`       // 任务运行结果
	Extra        json.RawMessage `json:"extra"`        // 扩展字段，用于未来功能扩展
	Common       json.RawMessage `json:"common"`       // 公共配置
	Specific     json.RawMessage `json:"specific"`     // 类型特有配置
	CreatedAt    time.Time       `json:"created_at"`
	UpdatedAt    time.Time       `json:"updated_at"`
	LastRunTime  *time.Time      `json:"last_run_time,omitempty"`
	User         string          `json:"user"`
}

// PDataCommonConfig 任务公共配置结构
type PDataCommonConfig struct {
	Summary      string              `json:"summary"`
	NeedEmail    bool                `json:"needEmail"`
	EmailSubject string              `json:"emailSubject"`
	Recipients   []string            `json:"recipients"`
	Schedule     PDataScheduleConfig `json:"schedule"`
}

// PDataScheduleConfig 定时任务配置
type PDataScheduleConfig struct {
	Enable bool               `json:"enable"`
	Type   string             `json:"type"` // daily, weekly, monthly
	Daily  PDataDailySchedule `json:"daily"`
}

// PDataDailySchedule 每日定时配置
type PDataDailySchedule struct {
	Hour   string `json:"hour"`
	Minute string `json:"minute"`
}

// PDataTaskResult 任务运行结果结构（示例）
type PDataTaskResult struct {
	Success      bool        `json:"success"`
	Message      string      `json:"message,omitempty"`
	Data         interface{} `json:"data,omitempty"`
	DurationMs   int64       `json:"duration_ms,omitempty"`   // 执行耗时（毫秒）
	RowsAffected int         `json:"rows_affected,omitempty"` // 影响行数（SQL任务）
	Output       string      `json:"output,omitempty"`        // 输出内容（Python任务）
}

// PDataInitCron 初始化定时任务调度器
func PDataInitCron() {
	cronScheduler = cron.New()
	cronScheduler.Start()
	PDataRestoreScheduledTasks()
}

// PDataGetTasks 获取任务列表
func PDataGetTasks(c *gin.Context) {
	var tasks []PDataTask
	result := db.Find(&tasks)
	if result.Error != nil {
		c.JSON(500, gin.H{"success": false, "message": "获取任务列表失败: " + result.Error.Error()})
		return
	}

	c.JSON(200, gin.H{"success": true, "data": tasks})
}

// PDataGetTask 获取单个任务详情
func PDataGetTask(c *gin.Context) {
	id := c.Param("id")
	var task PDataTask
	result := db.First(&task, "id = ?", id)
	if result.Error != nil {
		if errors.Is(result.Error, gorm.ErrRecordNotFound) {
			c.JSON(404, gin.H{"success": false, "message": "任务不存在"})
			return
		}
		c.JSON(500, gin.H{"success": false, "message": "获取任务失败: " + result.Error.Error()})
		return
	}

	c.JSON(200, gin.H{"success": true, "data": task})
}

// PDataCreateTask 创建新任务
func PDataCreateTask(c *gin.Context) {
	var req struct {
		TaskName string          `json:"taskName" binding:"required"`
		TaskType string          `json:"taskType" binding:"required,oneof=sql python report"`
		Common   json.RawMessage `json:"common" binding:"required"`
		Specific json.RawMessage `json:"specific" binding:"required"`
		Extra    json.RawMessage `json:"extra,omitempty"` // 允许创建时提供扩展信息
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"success": false, "message": "参数错误: " + err.Error()})
		return
	}

	// 生成任务ID
	taskID := fmt.Sprintf("PDTask_%d", time.Now().UnixNano())

	// 解析公共配置以处理定时任务
	var commonConfig PDataCommonConfig
	if err := json.Unmarshal(req.Common, &commonConfig); err != nil {
		c.JSON(400, gin.H{"success": false, "message": "公共配置格式错误: " + err.Error()})
		return
	}

	task := PDataTask{
		ID:           taskID,
		TaskName:     req.TaskName,
		TaskType:     req.TaskType,
		Status:       "pending",
		StatusDetail: "任务已创建，等待执行",
		Common:       req.Common,
		Specific:     req.Specific,
		Extra:        req.Extra,
		CreatedAt:    time.Now(),
		UpdatedAt:    time.Now(),
	}

	result := db.Create(&task)
	if result.Error != nil {
		c.JSON(500, gin.H{"success": false, "message": "创建任务失败: " + result.Error.Error()})
		return
	}

	// 如果启用了定时任务，添加到调度器
	if commonConfig.Schedule.Enable {
		if err := PDataAddTaskToCron(&task, &commonConfig.Schedule); err != nil {
			// 定时任务添加失败不影响任务创建，但记录错误
			db.Model(&task).Update("status_detail", fmt.Sprintf("任务创建成功，但定时配置失败: %v", err))
			c.JSON(200, gin.H{
				"success": true,
				"message": "任务创建成功，但定时配置失败",
				"data":    task,
			})
			return
		}
	}

	c.JSON(200, gin.H{"success": true, "data": task})
}

// PDataUpdateTask 更新任务
func PDataUpdateTask(c *gin.Context) {
	id := c.Param("id")
	var req struct {
		TaskName string          `json:"taskName" binding:"required"`
		TaskType string          `json:"taskType"`
		Common   json.RawMessage `json:"common" binding:"required"`
		Specific json.RawMessage `json:"specific" binding:"required"`
		Extra    json.RawMessage `json:"extra,omitempty"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"success": false, "message": "参数错误: " + err.Error()})
		return
	}

	// 检查任务是否存在
	var existingTask PDataTask
	result := db.First(&existingTask, "id = ?", id)
	if result.Error != nil {
		if errors.Is(result.Error, gorm.ErrRecordNotFound) {
			c.JSON(404, gin.H{"success": false, "message": "任务不存在"})
			return
		}
		c.JSON(500, gin.H{"success": false, "message": "查询任务失败: " + result.Error.Error()})
		return
	}

	// 解析公共配置以处理定时任务
	var commonConfig PDataCommonConfig
	if err := json.Unmarshal(req.Common, &commonConfig); err != nil {
		c.JSON(400, gin.H{"success": false, "message": "公共配置格式错误: " + err.Error()})
		return
	}

	// 先移除旧的定时任务
	if entryID, exists := taskEntryMap[id]; exists {
		cronScheduler.Remove(entryID)
		delete(taskEntryMap, id)
	}

	// 更新任务
	task := PDataTask{
		ID:           id,
		TaskName:     req.TaskName,
		TaskType:     req.TaskType,
		Status:       existingTask.Status,
		StatusDetail: existingTask.StatusDetail, // 保留状态详情
		Result:       existingTask.Result,       // 保留历史结果
		Common:       req.Common,
		Specific:     req.Specific,
		Extra:        req.Extra,
		UpdatedAt:    time.Now(),
		CreatedAt:    existingTask.CreatedAt,
		LastRunTime:  existingTask.LastRunTime,
	}

	result = db.Save(&task)
	if result.Error != nil {
		c.JSON(500, gin.H{"success": false, "message": "更新任务失败: " + result.Error.Error()})
		return
	}

	// 如果启用了定时任务，添加到调度器
	updateMsg := ""
	if commonConfig.Schedule.Enable {
		if err := PDataAddTaskToCron(&task, &commonConfig.Schedule); err != nil {
			updateMsg = "，但定时配置失败"
			db.Model(&task).Update("status_detail", fmt.Sprintf("任务更新成功，但定时配置失败: %v", err))
		}
	}

	c.JSON(200, gin.H{
		"success": true,
		"message": "任务更新成功" + updateMsg,
		"data":    task,
	})
}

// PDataDeleteTask 删除任务
func PDataDeleteTask(c *gin.Context) {
	id := c.Param("id")

	// 先移除定时任务
	if entryID, exists := taskEntryMap[id]; exists {
		cronScheduler.Remove(entryID)
		delete(taskEntryMap, id)
	}

	// 删除任务记录
	result := db.Delete(&PDataTask{}, "id = ?", id)
	if result.Error != nil {
		c.JSON(500, gin.H{"success": false, "message": "删除任务失败: " + result.Error.Error()})
		return
	}

	if result.RowsAffected == 0 {
		c.JSON(404, gin.H{"success": false, "message": "任务不存在"})
		return
	}

	c.JSON(200, gin.H{"success": true, "message": "任务删除成功"})
}

// PDataExecuteTask 立即执行任务
func PDataExecuteTask(c *gin.Context) {
	id := c.Param("id")
	log.Printf("收到执行任务请求，任务ID: %s，请求IP: %s", id, c.ClientIP())
	var task PDataTask
	log.Printf("开始查询任务信息，任务ID: %s", id)
	result := db.First(&task, "id = ?", id)
	if result.Error != nil {
		if errors.Is(result.Error, gorm.ErrRecordNotFound) {
			c.JSON(404, gin.H{"success": false, "message": "任务不存在"})
			return
		}
		log.Printf("查询任务失败，任务ID: %s，错误信息: %v", id, result.Error)
		c.JSON(500, gin.H{"success": false, "message": "获取任务失败: " + result.Error.Error()})
		return
	}
	log.Printf("任务查询成功，任务ID: %s，任务名称: %s，当前状态: %v", id, task.TaskName, task.Status)
	// 异步执行任务
	go func() {

		PDataRunTask(&task)
	}()

	c.JSON(200, gin.H{"success": true, "message": "任务已开始执行"})
}

// PDataAddTaskToCron 将任务添加到定时调度器
func PDataAddTaskToCron(task *PDataTask, schedule *PDataScheduleConfig) error {
	// 生成cron表达式
	var cronExpr string
	switch schedule.Type {
	case "daily":
		// 每日定时: 分 时 * * *
		cronExpr = fmt.Sprintf("%s %s * * *", schedule.Daily.Minute, schedule.Daily.Hour)
	default:
		cronExpr = fmt.Sprintf("%s %s * * *", schedule.Daily.Minute, schedule.Daily.Hour)
	}

	// 添加任务到调度器
	taskID := task.ID
	entryID, err := cronScheduler.AddFunc(cronExpr, func() {
		var latestTask PDataTask
		if err := db.First(&latestTask, "id = ?", taskID).Error; err == nil {
			PDataRunTask(&latestTask)
		} else {
			fmt.Printf("定时任务执行失败，获取任务 %s 信息错误: %v\n", taskID, err)
		}
	})

	if err != nil {
		return err
	}

	// 记录映射关系
	taskEntryMap[task.ID] = entryID
	return nil
}

// PDataRestoreScheduledTasks 服务启动时恢复所有定时任务
func PDataRestoreScheduledTasks() {
	var tasks []PDataTask
	result := db.Find(&tasks)
	if result.Error != nil {
		fmt.Printf("恢复定时任务失败: %v\n", result.Error)
		return
	}

	for _, task := range tasks {
		var commonConfig PDataCommonConfig
		if err := json.Unmarshal(task.Common, &commonConfig); err != nil {
			fmt.Printf("解析任务 %s 配置失败: %v\n", task.ID, err)
			continue
		}

		if commonConfig.Schedule.Enable {
			if err := PDataAddTaskToCron(&task, &commonConfig.Schedule); err != nil {
				fmt.Printf("恢复任务 %s 定时失败: %v\n", task.ID, err)
				// 更新任务状态详情
				db.Model(&PDataTask{}).Where("id = ?", task.ID).Update(
					"status_detail", fmt.Sprintf("服务重启后定时任务恢复失败: %v", err),
				)
			} else {
				fmt.Printf("恢复任务 %s 定时成功\n", task.ID)
				db.Model(&PDataTask{}).Where("id = ?", task.ID).Update(
					"status_detail", "服务重启后定时任务已恢复",
				)
			}
		}
	}
}

// PDataRunTask 执行任务的实际逻辑
func PDataRunTask(task *PDataTask) {
	// 更新任务状态为运行中
	startTime := time.Now()
	updateData := map[string]interface{}{
		"status":        "running",
		"status_detail": fmt.Sprintf("任务开始执行 at %v", startTime.Format(time.RFC3339)),
	}
	db.Model(&PDataTask{}).Where("id = ?", task.ID).Updates(updateData)

	fmt.Printf("任务 %s 开始执行: %v\n", task.ID, startTime)

	// 执行结果初始化
	result := PDataTaskResult{
		Success:    false,
		DurationMs: 0,
	}
	var finalStatus string = "failed"
	var err error

	// 根据任务类型执行不同的逻辑
	switch task.TaskType {
	case "sql":
		result.Data, err = PDataExecuteSQLTask(task)
	case "python":
		result.Data, err = PDataExecutePythonTask(task)
	case "report":
		result.Data, err = PDataExecuteReportTask(task)
	default:
		err = fmt.Errorf("未知任务类型: %s", task.TaskType)
	}

	// 计算执行耗时
	duration := time.Since(startTime)
	result.DurationMs = duration.Milliseconds()

	// 处理执行结果
	if err != nil {
		result.Message = err.Error()
		finalStatus = "failed"
		log.Printf("任务执行失败: %s %v (耗时: %v)", err, err, duration)
		statusDetail := fmt.Sprintf("任务执行失败: %s %v (耗时: %v)", err, err, duration)
		fmt.Printf("任务 %s %s\n", task.ID, statusDetail)

		// 更新任务状态和详情
		resultJSON, _ := json.Marshal(result)
		now := time.Now()
		db.Model(&PDataTask{}).Where("id = ?", task.ID).Updates(map[string]interface{}{
			"status":        finalStatus,
			"status_detail": statusDetail,
			"result":        resultJSON,
			"last_run_time": &now,
			"updated_at":    now,
		})
	} else {
		result.Success = true
		result.Message = "任务执行成功"
		finalStatus = "completed"
		statusDetail := fmt.Sprintf("任务执行成功 (耗时: %v)", duration)
		fmt.Printf("任务 %s %s\n", task.ID, statusDetail)

		// 更新任务状态和详情
		resultJSON, _ := json.Marshal(result)
		now := time.Now()
		db.Model(&PDataTask{}).Where("id = ?", task.ID).Updates(map[string]interface{}{
			"status":        finalStatus,
			"status_detail": statusDetail,
			"result":        resultJSON,
			"last_run_time": &now,
			"updated_at":    now,
		})
	}

	// 发送邮件通知
	var commonConfig PDataCommonConfig
	if err := json.Unmarshal(task.Common, &commonConfig); err == nil && commonConfig.NeedEmail {
		PDataSendTaskNotification(task, finalStatus, &commonConfig, &result)
	}
}

// PDataExecuteSQLTask 执行SQL任务
func PDataExecuteSQLTask(task *PDataTask) (interface{}, error) {
	// 解析SQL任务配置
	var sqlConfig struct {
		SQLContent      string `json:"sqlContent"`
		DataDescription string `json:"dataDescription"`
	}

	if err := json.Unmarshal(task.Specific, &sqlConfig); err != nil {
		return nil, fmt.Errorf("解析SQL任务配置失败: %v", err)
	}

	if sqlConfig.SQLContent == "" {
		return nil, errors.New("SQL语句不能为空")
	}

	fmt.Printf("执行SQL任务 %s: %s\n", task.ID, sqlConfig.SQLContent)

	// 实际执行SQL的逻辑（示例）
	// 这里应该是实际执行SQL查询的代码
	// rows, err := db.Raw(sqlConfig.SQLContent).Rows()
	// ... 处理查询结果

	// 模拟执行结果
	return map[string]interface{}{
		"query": sqlConfig.SQLContent,
		// "columns": []string{"id", "name", "value"},
		// "rows": [...]
		"message": "SQL查询执行成功（模拟结果）",
	}, nil
}

// PDataExecutePythonTask 执行Python任务
func PDataExecutePythonTask(task *PDataTask) (interface{}, error) {
	// 解析Python任务配置
	var pythonConfig struct {
		Code        string            `json:"code"`
		InputParams map[string]string `json:"inputParams"`
		OutputType  string            `json:"outputType"`
	}

	if err := json.Unmarshal(task.Specific, &pythonConfig); err != nil {
		return nil, fmt.Errorf("解析Python任务配置失败: %v", err)
	}

	if pythonConfig.Code == "" {
		return nil, errors.New("Python代码不能为空")
	}

	fmt.Printf("执行Python任务 %s，参数: %v\n", task.ID, pythonConfig.InputParams)

	// 实际执行Python代码的逻辑（示例）
	// 这里应该是调用Python解释器执行代码的逻辑
	// output, err := execPythonCode(pythonConfig.Code, pythonConfig.InputParams)

	// 模拟执行结果
	return map[string]interface{}{
		"params": pythonConfig.InputParams,
		"output": "Python脚本执行成功（模拟输出）",
		"type":   pythonConfig.OutputType,
	}, nil
}

// // PDataExecuteReportTask 执行邮件任务
// func PDataExecuteReportTask(task *PDataTask) (interface{}, error) {
// 	// 解析邮件任务配置
// 	var reportConfig struct {
// 		BodySections []struct {
// 			Title   string `json:"title"`
// 			Content string `json:"content"`
// 			SQL     string `json:"sql"`
// 		} `json:"bodySections"`
// 	}
// 	log.Printf("task.Specific:", task.Specific)

// 	if err := json.Unmarshal(task.Specific, &reportConfig); err != nil {
// 		return nil, fmt.Errorf("解析邮件任务配置失败: %v", err)
// 	}
// 	log.Printf("BodySections:", reportConfig.BodySections)
// 	if len(reportConfig.BodySections) == 0 {
// 		return nil, errors.New("邮件内容段落不能为空")
// 	}

// 	// 检查是否有段落既没有内容也没有SQL
// 	for i, section := range reportConfig.BodySections {
// 		if section.Content == "" && section.SQL == "" {
// 			return nil, fmt.Errorf("段落 %d 必须填写内容或SQL语句", i+1)
// 		}
// 	}

// 	fmt.Printf("执行邮件任务 %s，包含 %d 个段落\n", task.ID, len(reportConfig.BodySections))

// 	// 实际执行邮件任务的逻辑（示例）
// 	// 1. 执行各段落的SQL获取数据
// 	// 2. 生成邮件内容
// 	// 3. 发送邮件

// 	// 模拟执行结果
// 	return map[string]interface{}{
// 		"sections_count": len(reportConfig.BodySections),
// 		"message":        "邮件已发送（模拟结果）",
// 	}, nil
// }

// PDataSendTaskNotification 发送任务执行结果通知
func PDataSendTaskNotification(task *PDataTask, status string, config *PDataCommonConfig, result *PDataTaskResult) {
	if len(config.Recipients) == 0 {
		fmt.Printf("任务 %s 通知失败: 收件人列表为空\n", task.ID)
		return
	}

	// 构建邮件内容
	subject := fmt.Sprintf("%s - %s %s", config.EmailSubject, task.TaskName, status)
	content := fmt.Sprintf("任务 %s 执行结果：\n状态：%s\n详情：%s\n耗时：%d ms",
		task.TaskName,
		status,
		result.Message,
		result.DurationMs,
	)

	// 实际发送邮件的逻辑（示例）
	fmt.Printf("发送邮件通知:\n收件人: %v\n主题: %s\n内容: %s\n",
		config.Recipients, subject, content)

	// 更新任务的扩展字段，记录最后一次通知时间
	extraData := make(map[string]interface{})
	if task.Extra != nil {
		json.Unmarshal(task.Extra, &extraData) // 忽略错误，处理已有数据
	}
	extraData["last_notify_time"] = time.Now().Format(time.RFC3339)
	extraJSON, _ := json.Marshal(extraData)

	db.Model(&PDataTask{}).Where("id = ?", task.ID).Update("extra", extraJSON)
}

// PDataSetupRoutes 配置任务相关路由
func PDataSetupRoutes(r *gin.RouterGroup) {
	taskRoutes := r.Group("/pdtasks")
	{
		taskRoutes.GET("", authMiddleware, PDataGetTasks)
		taskRoutes.GET("/:id", authMiddleware, PDataGetTask)
		taskRoutes.POST("", authMiddleware, PDataCreateTask)
		taskRoutes.PUT("/:id", authMiddleware, PDataUpdateTask)
		taskRoutes.DELETE("/:id", authMiddleware, PDataDeleteTask)
		taskRoutes.POST("/:id/execute", authMiddleware, PDataExecuteTask)
	}
}


type ReportTemplate struct {
	ID          uint      `gorm:"primaryKey" json:"id"`
	Username    string    `gorm:"size:50;not null;index:idx_user_report" json:"username"` // 关联用户
	ReportName  string    `gorm:"size:100;not null;index:idx_user_report" json:"report_name"` // 报表名称（用户下唯一）
	TemplateJSON string   `gorm:"type:text;not null" json:"template_json"` // 模板完整JSON配置
	Description string    `gorm:"size:500" json:"description"` // 模板描述
	FilePath    string    `gorm:"size:255" json:"file_path"` // 关联的CSV数据文件路径
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
	DeletedAt   *time.Time `gorm:"index" json:"-"` // 软删除字段
}

// TableName 自定义表名
func (ReportTemplate) TableName() string {
	return "report_templates"
}

