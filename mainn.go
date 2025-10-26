package main

import (
	"bufio"
	"bytes"
	"embed"
	"regexp"

	// 	"crypto"
	"crypto/aes"
	"crypto/cipher"

	"path/filepath"
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
	"golang.org/x/crypto/bcrypt"
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
	// 	"gonum.org/v1/gonum/stat/classif"
	// 矩阵操作（仍在主模块）
	// 辅助统计函数
	// 	"github.com/sjwhitworth/golearn/base"
	// 	"github.com/sjwhitworth/golearn/ensemble"
	// 	"github.com/sjwhitworth/golearn/trees"
	//      "github.com/sjwhitworth/golearn/base"
	// 	"github.com/go-gota/gota/dataframe"  // 类似 Pandas 的 DataFrame
	//     "github.com/go-gota/gota/series"    // 数据处理
	//     "github.com/montanaflynn/stats"     // 统计分析
	//     "github.com/sajari/regression"      // 回归分析
)

var jwtSecret = []byte("your-secret-key") // 从环境变量读取更安全
type Claims struct {
	Username string `json:"username"`
	Role     string `json:"role"`
	jwt.RegisteredClaims
}

//go:embed all:web/out
//go:embed all:config
var resources embed.FS // 用 embed.FS 类型的变量接收嵌入的资源
// 全局变量：资源释放目录
var releaseDir string

var loggedInUsers = make(map[string]bool) // 全局已登录用户存储

type NovaSSOConfig struct {
	SSOUrl        string    `json:"ssoUrl"`
	Method        string    `json:"method"`
	UserField     string    `json:"userField"`
	PasswordField string    `json:"passwordField"`
	SsoSussJudge  string    `json:"ssoSussJudge"`
	UrlEncode     bool      `json:"urlEncode"`
	Checksum      string    `gorm:"column:checksum"`
	Timestamp     time.Time `gorm:"column:timestamp"`
}

func validateSSOLogin(username, password string) error {
	// 查询 SSO 配置
	var ssoConfig NovaSSOConfig
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

func valid_db_password(db *gorm.DB, username, password string, n int) (*NovaUser, error) {
	var user NovaUser

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

type NovaUser struct {
	ID          uint            `gorm:"primaryKey"`
	Username    string          `gorm:"column:username;type:varchar(255);uniqueIndex"` // 用户名不加密
	Email       string          `gorm:"column:email"`                                  // 邮箱不加密
	Telephone   string          `gorm:"column:telephone"`                              // 邮箱不加密
	Company     string          `gorm:"column:company"`                                // 邮箱不加密
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

func (u *NovaUser) BeforeSave(tx *gorm.DB) error {
	// 计算用户名的哈希值
	data := []byte(fmt.Sprintf("%s%s%s%s", u.Username, u.Role, u.GroupName, u.Password))
	u.Checksum = calculateChecksum(data)
	u.Timestamp = time.Now()
	return nil
}

// AfterFind 在读取用户数据后验证校验和
func (u *NovaUser) AfterFind(tx *gorm.DB) error {
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

	c.Next()
}
func releaseResources(targetDir string) error {
	// 创建目标目录
	if err := os.MkdirAll(targetDir, 0755); err != nil {
		return fmt.Errorf("创建目录失败：%w", err)
	}

	// 释放前端资源到 targetDir/build
	frontendFS, err := fs.Sub(resources, "web/out")
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

	// filePath := "datafactory_fetchdata.py"
	// if _, err := os.Stat(filePath); os.IsNotExist(err) {
	// 	// 文件不存在，才进行释放
	// 	content, err := fs.ReadFile(resources, filePath)
	// 	if err != nil {
	// 		return fmt.Errorf("提取配置资源失败：%w", err)
	// 	}

	// 	if err := os.WriteFile(filePath, content, 0644); err != nil {
	// 		return fmt.Errorf("释放脚本文件失败：%w", err)
	// 	}
	// } else if err != nil {
	// 	// 其他错误情况
	// 	return fmt.Errorf("检查文件是否存在失败：%w", err)
	// }
	// filePath2 := "genCustomReport.py"
	// if _, err := os.Stat(filePath); os.IsNotExist(err) {
	// 	// 文件不存在，才进行释放
	// 	content, err := fs.ReadFile(resources, filePath2)
	// 	if err != nil {
	// 		return fmt.Errorf("提取配置资源失败：%w", err)
	// 	}

	// 	if err := os.WriteFile(filePath2, content, 0644); err != nil {
	// 		return fmt.Errorf("释放脚本文件失败：%w", err)
	// 	}
	// } else if err != nil {
	// 	// 其他错误情况
	// 	return fmt.Errorf("检查文件是否存在失败：%w", err)
	// }

	// if err := os.WriteFile("datafactory_fetchdata.py", content, 0644); err != nil {
	// 	return fmt.Errorf("释放脚本文件失败：%w", err)
	// }

	return nil
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
		db, err = gorm.Open(sqlite.Open("nova.db"), &gorm.Config{})
		if err != nil {
			return nil, fmt.Errorf("无法连接 SQLite 数据库: %v", err)
		}
		logger.Println("已连接到 SQLite 数据库")
	}

	// 自动迁移表结构
	if err := db.AutoMigrate(&NovaUser{},
		&ForumPost{},
		&ForumCategory{},
		&ForumTag{},
		&UserPostInteraction{},
	); err != nil {
		return nil, fmt.Errorf("自动迁移表结构失败: %v", err)
	}

	if err := db.AutoMigrate(&NovaSSOConfig{}); err != nil {
		return nil, fmt.Errorf("自动迁移表结构失败: %v", err)
	}

	return db, nil
}
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

var (
	db     *gorm.DB
	logger = logrus.New()
	// saEngine           *gorm.DB  // 模拟 Flask 中的 sa_engine
	cfg                *ini.File // 全局配置文件对象
	defaultOptionsData map[string]interface{}
	de_engine          *gorm.DB
	run_mode           string

	// 任务ID与定时任务EntryID的映射，用于更新和删除定时任务
	// 重命名变量避免与包名冲突

	server_ip   string
	server_port string
)

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
	var user NovaUser
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
func isPortAvailable(port int) bool {
	listener, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return false
	}
	defer listener.Close()
	return true
}
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

func main() {
	releaseDir := "."
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
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
		c.Writer.Header().Set("Access-Control-Allow-Credentials", "true")                       // 允许携带凭证
		c.Writer.Header().Set("Access-Control-Expose-Headers", "Authorization, Content-Length") // 添加这一行
		c.Writer.Header().Set("Access-Control-Max-Age", "86400")                                // 添加预检请求缓存
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
	r.POST("/auth/register", RegisterHandler)

	setupRoutes(r)

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

type ForumPost struct {
	ID           uint      `json:"id" gorm:"primaryKey"`
	Title        string    `json:"title" gorm:"size:255;not null"`
	Content      string    `json:"content" gorm:"type:text;not null"`
	Excerpt      string    `json:"excerpt" gorm:"size:500"`
	AuthorID     uint      `json:"author_id" gorm:"not null"`
	Author       string    `json:"author"`
	Category     string    `json:"category" gorm:"size:50;not null;default:'general'"`
	Tags         string    `json:"tags" gorm:"type:text"` // JSON数组存储
	Likes        int       `json:"likes" gorm:"default:0"`
	Comments     int       `json:"comments" gorm:"default:0"`
	Views        int       `json:"views" gorm:"default:0"`
	IsLiked      bool      `json:"is_liked" gorm:"-"`
	IsBookmarked bool      `json:"is_bookmarked" gorm:"-"`
	IsPinned     bool      `json:"is_pinned" gorm:"default:false"`
	IsFeatured   bool      `json:"is_featured" gorm:"default:false"`
	ReadingTime  int       `json:"reading_time" gorm:"default:0"`
	Difficulty   string    `json:"difficulty" gorm:"size:20;default:'intermediate'"`
	LastActivity time.Time `json:"last_activity" gorm:"autoUpdateTime"`
	CreatedAt    time.Time `json:"created_at" gorm:"autoCreateTime"`
	UpdatedAt    time.Time `json:"updated_at" gorm:"autoUpdateTime"`
}

// 论坛分类
type ForumCategory struct {
	ID          uint   `json:"id" gorm:"primaryKey"`
	Name        string `json:"name" gorm:"size:100;not null;unique"`
	Description string `json:"description" gorm:"size:255"`
	Color       string `json:"color" gorm:"size:20;default:'primary'"`
	Icon        string `json:"icon" gorm:"size:50"`
	SortOrder   int    `json:"sort_order" gorm:"default:0"`
	IsActive    bool   `json:"is_active" gorm:"default:true"`
}

// 帖子标签
type ForumTag struct {
	ID    uint   `json:"id" gorm:"primaryKey"`
	Name  string `json:"name" gorm:"size:50;not null;unique"`
	Count int    `json:"count" gorm:"default:0"`
	Trend string `json:"trend" gorm:"size:10;default:'stable'"`
}

// 用户帖子交互（点赞、收藏）
type UserPostInteraction struct {
	ID         uint      `json:"id" gorm:"primaryKey"`
	UserID     uint      `json:"user_id" gorm:"not null"`
	PostID     uint      `json:"post_id" gorm:"not null"`
	Liked      bool      `json:"liked" gorm:"default:false"`
	Bookmarked bool      `json:"bookmarked" gorm:"default:false"`
	CreatedAt  time.Time `json:"created_at" gorm:"autoCreateTime"`
	UpdatedAt  time.Time `json:"updated_at" gorm:"autoUpdateTime"`
}

// 论坛统计
type ForumStats struct {
	TotalPosts     int `json:"total_posts"`
	TotalComments  int `json:"total_comments"`
	TotalLikes     int `json:"total_likes"`
	TotalMembers   int `json:"total_members"`
	OnlineMembers  int `json:"online_members"`
	TodayPosts     int `json:"today_posts"`
	TrendingTopics int `json:"trending_topics"`
}

// API 请求/响应结构体
type CreatePostRequest struct {
	Title    string   `json:"title" binding:"required"`
	Content  string   `json:"content" binding:"required"`
	Excerpt  string   `json:"excerpt"` // 添加这个字段
	Category string   `json:"category" `
	Tags     []string `json:"tags"`

	IsPublic bool `json:"is_public" default:"true"`
}

type UpdatePostRequest struct {
	Title    string   `json:"title"`
	Content  string   `json:"content"`
	Category string   `json:"category"`
	Tags     []string `json:"tags"`
}

type PostResponse struct {
	ForumPost
	IsLiked      bool `json:"is_liked"`
	IsBookmarked bool `json:"is_bookmarked"`
}

type PostsListResponse struct {
	Posts      []PostResponse `json:"posts"`
	Total      int64          `json:"total"`
	Page       int            `json:"page"`
	TotalPages int            `json:"total_pages"`
	Stats      ForumStats     `json:"stats"`
}

type ToggleInteractionRequest struct {
	PostID uint `json:"post_id" binding:"required"`
}

func setupRoutes(router *gin.Engine) {
	// ... 现有的路由 ...

	// 论坛相关路由
	forum := router.Group("/api/forum")
	{
		forum.GET("/posts", getForumPosts)
		forum.GET("/posts/:id", getPostDetail)
		forum.POST("/posts", authMiddleware, createPost)
		forum.PUT("/posts/:id", authMiddleware, updatePost)
		forum.DELETE("/posts/:id", authMiddleware, deletePost)

		forum.POST("/posts/:id/like", authMiddleware, toggleLike)
		forum.POST("/posts/:id/bookmark", authMiddleware, toggleBookmark)
        forum.POST("/posts/upload", authMiddleware, uploadFile)
        forum.GET("/posts/:id/attachments", getPostAttachments)
        forum.DELETE("/attachments/:id", authMiddleware, deleteAttachment)


		forum.GET("/categories", getCategories)
		forum.GET("/tags/trending", getTrendingTags)
		forum.GET("/stats", getForumStats)

		// 管理接口
		admin := forum.Group("/admin")
		admin.Use(authMiddleware)
		{
			admin.POST("/categories", createCategory)
			admin.PUT("/categories/:id", updateCategory)
			admin.DELETE("/categories/:id", deleteCategory)
			admin.POST("/posts/:id/pin", togglePinPost)
			admin.POST("/posts/:id/feature", toggleFeaturePost)
		}



	}
}

// 获取论坛帖子列表
func getForumPosts(c *gin.Context) {
	logger.Info("Entered getForumPosts handler")

	// 解析查询参数
	page, _ := strconv.Atoi(c.DefaultQuery("page", "1"))
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "10"))
	category := c.DefaultQuery("category", "all")
	sort := c.DefaultQuery("sort", "latest")
	search := c.Query("search")

	if page < 1 {
		page = 1
	}
	if limit < 1 || limit > 50 {
		limit = 10
	}

	offset := (page - 1) * limit

	// 构建查询 - 移除 Preload("Author")
	query := db.Model(&ForumPost{})

	// 分类筛选
	if category != "all" {
		query = query.Where("category = ?", category)
	}

	// 搜索
	if search != "" {
		searchLike := "%" + search + "%"
		query = query.Where("title LIKE ? OR excerpt LIKE ? OR content LIKE ?", searchLike, searchLike, searchLike)
	}

	// 排序
	switch sort {
	case "latest":
		query = query.Order("created_at DESC")
	case "hot":
		query = query.Order("(likes + comments * 2) DESC")
	case "trending":
		query = query.Order("(views / 100 + likes + comments) DESC")
	case "most_liked":
		query = query.Order("likes DESC")
	default:
		query = query.Order("created_at DESC")
	}

	// 获取总数
	var total int64
	if err := query.Count(&total).Error; err != nil {
		logger.Errorf("Failed to count posts: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取帖子数量失败"})
		return
	}

	// 获取帖子列表
	var posts []ForumPost
	if err := query.Offset(offset).Limit(limit).Find(&posts).Error; err != nil {
		logger.Errorf("Failed to fetch posts: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取帖子列表失败"})
		return
	}

	// 简化：直接返回帖子列表，不处理用户交互状态
	responsePosts := make([]PostResponse, len(posts))
	for i, post := range posts {
		responsePosts[i] = PostResponse{
			ForumPost:    post,
			IsLiked:      false, // 简化：默认未点赞
			IsBookmarked: false, // 简化：默认未收藏
		}
	}

	// 获取统计信息
	stats := getForumStatsInternal()

	c.JSON(http.StatusOK, PostsListResponse{
		Posts:      responsePosts,
		Total:      total,
		Page:       page,
		TotalPages: int((total + int64(limit) - 1) / int64(limit)),
		Stats:      stats,
	})
}

// 获取帖子详情
func getPostDetail(c *gin.Context) {
	logger.Info("Entered getPostDetail handler")

	postID, err := strconv.Atoi(c.Param("id"))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "无效的帖子ID"})
		return
	}

	var post ForumPost
	// 移除 Preload("Author")
	if err := db.First(&post, postID).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			c.JSON(http.StatusNotFound, gin.H{"error": "帖子不存在"})
			return
		}
		logger.Errorf("Failed to fetch post: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取帖子失败"})
		return
	}

	// 增加浏览量
	db.Model(&post).UpdateColumn("views", gorm.Expr("views + ?", 1))
	post.Views++

	// 简化：移除用户交互状态检查
	post.IsLiked = false
	post.IsBookmarked = false

	c.JSON(http.StatusOK, post)
}

// 创建帖子
func createPost(c *gin.Context) {
	logger.Info("Entered createPost handler")

	// 获取当前用户
	user, ok := getCurrentUser(c) //currentUser.(User)
	if !ok {
		logger.Warn("Failed to assert current user type")
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "内部服务器错误"})
		return
	}
	username := user.Username // 或从JWT中获取
	fmt.Printf("in createPost username:%s", username)

     if !isUserAuthorized(username) {
        c.JSON(http.StatusForbidden, gin.H{"error": "用户未认证，无法发布帖子"})
        return
    }

	var req CreatePostRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		logger.Errorf("Failed to bind JSON: %v", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": "无效的请求数据"})
		return
	}

	// 计算阅读时间（按中文字符数估算）
	readingTime := len([]rune(req.Content)) / 500
	if readingTime < 1 {
		readingTime = 1
	}

	// 生成摘要
	excerpt := req.Excerpt
	if excerpt == "" {
		runes := []rune(req.Content)
		if len(runes) > 150 {
			excerpt = string(runes[:150]) + "..."
		} else {
			excerpt = req.Content
		}
	}

	// 创建帖子
	post := ForumPost{
		Title:   req.Title,
		Content: req.Content,
		Author:  username,
		// AuthorID: user.ID,

		//Tags:        strings.Join(req.Tags, ","),
		ReadingTime: readingTime,

		LastActivity: time.Now(),
	}

	if err := db.Create(&post).Error; err != nil {
		logger.Errorf("Failed to create post: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "创建帖子失败"})
		return
	}

	logger.Infof("Post created successfully: %d", post.ID)
	c.JSON(http.StatusCreated, gin.H{
		"status":  "success",
		"message": "帖子创建成功",
		"post_id": post.ID,
	})
}

// 更新帖子
func updatePost(c *gin.Context) {
	logger.Info("Entered updatePost handler")

	postID, err := strconv.Atoi(c.Param("id"))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "无效的帖子ID"})
		return
	}

	var req UpdatePostRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		logger.Errorf("Failed to bind JSON: %v", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": "无效的请求数据"})
		return
	}

	// 获取当前用户
	user, ok := getCurrentUser(c) //currentUser.(User)
	if !ok {
		logger.Warn("Failed to assert current user type")
		c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "message": "未登录"})
		return
	}
	username := user.Username // 或从JWT中获取


	var user NovaUser
	if err := db.Where("username = ?", username).First(&user).Error; err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "用户不存在"})
		return
	}

	// 检查帖子是否存在且属于当前用户
	var post ForumPost
	if err := db.First(&post, postID).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			c.JSON(http.StatusNotFound, gin.H{"error": "帖子不存在"})
			return
		}
		logger.Errorf("Failed to fetch post: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取帖子失败"})
		return
	}
    if post.Author != user.Username {
        c.JSON(http.StatusForbidden, gin.H{"error": "无权操作此帖子"})
        return
    }

	if post.AuthorID != user.ID {
		c.JSON(http.StatusForbidden, gin.H{"error": "无权修改此帖子"})
		return
	}

	// 更新帖子
	updates := make(map[string]interface{})
	if req.Title != "" {
		updates["title"] = req.Title
	}
	if req.Content != "" {
		updates["content"] = req.Content
		// 重新计算阅读时间
		readingTime := len([]rune(req.Content)) / 500
		if readingTime < 1 {
			readingTime = 1
		}
		updates["reading_time"] = readingTime
	}
	if req.Category != "" {
		updates["category"] = req.Category
	}
	if req.Tags != nil {
		updates["tags"] = strings.Join(req.Tags, ",")
	}

	updates["last_activity"] = time.Now()

	if err := db.Model(&post).Updates(updates).Error; err != nil {
		logger.Errorf("Failed to update post: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "更新帖子失败"})
		return
	}

	logger.Infof("Post updated successfully: %d", post.ID)
	c.JSON(http.StatusOK, gin.H{
		"status":  "success",
		"message": "帖子更新成功",
	})
}

// 删除帖子
func deletePost(c *gin.Context) {
	logger.Info("Entered deletePost handler")

	postID, err := strconv.Atoi(c.Param("id"))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "无效的帖子ID"})
		return
	}

	// 获取当前用户
	username, exists := c.Get("username")
	if !exists {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "未登录"})
		return
	}

	var user NovaUser
	if err := db.Where("username = ?", username).First(&user).Error; err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "用户不存在"})
		return
	}

	// 检查帖子是否存在且属于当前用户
	var post ForumPost
	if err := db.First(&post, postID).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			c.JSON(http.StatusNotFound, gin.H{"error": "帖子不存在"})
			return
		}
		logger.Errorf("Failed to fetch post: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取帖子失败"})
		return
	}

	if post.AuthorID != user.ID {
		c.JSON(http.StatusForbidden, gin.H{"error": "无权删除此帖子"})
		return
	}

	// 删除帖子
	if err := db.Delete(&post).Error; err != nil {
		logger.Errorf("Failed to delete post: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "删除帖子失败"})
		return
	}

	logger.Infof("Post deleted successfully: %d", post.ID)
	c.JSON(http.StatusOK, gin.H{
		"status":  "success",
		"message": "帖子删除成功",
	})
}

// 点赞/取消点赞
func toggleLike(c *gin.Context) {
	logger.Info("Entered toggleLike handler")

	postID, err := strconv.Atoi(c.Param("id"))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "无效的帖子ID"})
		return
	}

	// 获取当前用户
	username, exists := c.Get("username")
	if !exists {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "未登录"})
		return
	}

	var user NovaUser
	if err := db.Where("username = ?", username).First(&user).Error; err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "用户不存在"})
		return
	}

	// 检查帖子是否存在
	var post ForumPost
	if err := db.First(&post, postID).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			c.JSON(http.StatusNotFound, gin.H{"error": "帖子不存在"})
			return
		}
		logger.Errorf("Failed to fetch post: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取帖子失败"})
		return
	}

	// 查找或创建用户交互记录
	var interaction UserPostInteraction
	err = db.Where("user_id = ? AND post_id = ?", user.ID, post.ID).First(&interaction).Error

	if err == gorm.ErrRecordNotFound {
		// 创建新的点赞记录
		interaction = UserPostInteraction{
			UserID: user.ID,
			PostID: post.ID,
			Liked:  true,
		}
		if err := db.Create(&interaction).Error; err != nil {
			logger.Errorf("Failed to create like: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "点赞失败"})
			return
		}
		// 增加帖子点赞数
		db.Model(&post).UpdateColumn("likes", gorm.Expr("likes + ?", 1))
	} else if err == nil {
		// 切换点赞状态
		newLikedState := !interaction.Liked
		if err := db.Model(&interaction).Update("liked", newLikedState).Error; err != nil {
			logger.Errorf("Failed to update like: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "更新点赞状态失败"})
			return
		}
		// 更新帖子点赞数
		if newLikedState {
			db.Model(&post).UpdateColumn("likes", gorm.Expr("likes + ?", 1))
		} else {
			db.Model(&post).UpdateColumn("likes", gorm.Expr("likes - ?", 1))
		}
	} else {
		logger.Errorf("Failed to check like status: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "检查点赞状态失败"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status": "success",
		"liked":  !interaction.Liked, // 返回新的状态
	})
}

// 收藏/取消收藏
func toggleBookmark(c *gin.Context) {
	logger.Info("Entered toggleBookmark handler")

	postID, err := strconv.Atoi(c.Param("id"))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "无效的帖子ID"})
		return
	}

	// 获取当前用户
	username, exists := c.Get("username")
	if !exists {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "未登录"})
		return
	}

	var user NovaUser
	if err := db.Where("username = ?", username).First(&user).Error; err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "用户不存在"})
		return
	}

	// 检查帖子是否存在
	var post ForumPost
	if err := db.First(&post, postID).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			c.JSON(http.StatusNotFound, gin.H{"error": "帖子不存在"})
			return
		}
		logger.Errorf("Failed to fetch post: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取帖子失败"})
		return
	}

	// 查找或创建用户交互记录
	var interaction UserPostInteraction
	err = db.Where("user_id = ? AND post_id = ?", user.ID, post.ID).First(&interaction).Error

	if err == gorm.ErrRecordNotFound {
		// 创建新的收藏记录
		interaction = UserPostInteraction{
			UserID:     user.ID,
			PostID:     post.ID,
			Bookmarked: true,
		}
		if err := db.Create(&interaction).Error; err != nil {
			logger.Errorf("Failed to create bookmark: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "收藏失败"})
			return
		}
	} else if err == nil {
		// 切换收藏状态
		newBookmarkedState := !interaction.Bookmarked
		if err := db.Model(&interaction).Update("bookmarked", newBookmarkedState).Error; err != nil {
			logger.Errorf("Failed to update bookmark: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "更新收藏状态失败"})
			return
		}
	} else {
		logger.Errorf("Failed to check bookmark status: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "检查收藏状态失败"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":     "success",
		"bookmarked": !interaction.Bookmarked, // 返回新的状态
	})
}

// 获取分类列表
func getCategories(c *gin.Context) {
	logger.Info("Entered getCategories handler")

	var categories []ForumCategory
	if err := db.Where("is_active = ?", true).Order("sort_order ASC").Find(&categories).Error; err != nil {
		logger.Errorf("Failed to fetch categories: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取分类失败"})
		return
	}

	c.JSON(http.StatusOK, categories)
}

// 获取热门标签
func getTrendingTags(c *gin.Context) {
	logger.Info("Entered getTrendingTags handler")

	var tags []ForumTag
	if err := db.Order("count DESC").Limit(20).Find(&tags).Error; err != nil {
		logger.Errorf("Failed to fetch trending tags: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取热门标签失败"})
		return
	}

	c.JSON(http.StatusOK, tags)
}

// 获取论坛统计
func getForumStats(c *gin.Context) {
	logger.Info("Entered getForumStats handler")

	stats := getForumStatsInternal()
	c.JSON(http.StatusOK, stats)
}

// 内部函数：获取论坛统计
func getForumStatsInternal() ForumStats {
	var totalPosts int64
	// var totalComments int64
	// var totalLikes int64
	var totalMembers int64
	var todayPosts int64

	db.Model(&ForumPost{}).Count(&totalPosts)
	db.Model(&NovaUser{}).Count(&totalMembers)

	// 今日帖子数
	today := time.Now().Format("2006-01-02")
	db.Model(&ForumPost{}).Where("DATE(created_at) = ?", today).Count(&todayPosts)

	// 获取总评论数和点赞数（这里需要根据你的评论表结构调整）
	// 暂时使用帖子表中的统计字段
	var likesSum struct{ Sum int }
	var commentsSum struct{ Sum int }
	db.Model(&ForumPost{}).Select("SUM(likes) as sum").Scan(&likesSum)
	db.Model(&ForumPost{}).Select("SUM(comments) as sum").Scan(&commentsSum)

	// 在线用户数（简单实现，实际应该用Redis等）
	onlineMembers := len(loggedInUsers)

	return ForumStats{
		TotalPosts:     int(totalPosts),
		TotalComments:  int(commentsSum.Sum),
		TotalLikes:     int(likesSum.Sum),
		TotalMembers:   int(totalMembers),
		OnlineMembers:  onlineMembers,
		TodayPosts:     int(todayPosts),
		TrendingTopics: 0, // 可根据业务逻辑实现
	}
}

// 更新标签计数
func updateTagCount(tagName string, increment int) {
	var tag ForumTag
	if err := db.Where("name = ?", tagName).First(&tag).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			// 创建新标签
			tag = ForumTag{
				Name:  tagName,
				Count: increment,
			}
			db.Create(&tag)
		}
	} else {
		// 更新现有标签计数
		db.Model(&tag).UpdateColumn("count", gorm.Expr("count + ?", increment))
	}
}

func createCategory(c *gin.Context) {
	logger.Info("Entered createCategory handler")

	var category ForumCategory
	if err := c.ShouldBindJSON(&category); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "无效的请求数据"})
		return
	}

	if err := db.Create(&category).Error; err != nil {
		logger.Errorf("Failed to create category: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "创建分类失败"})
		return
	}

	c.JSON(http.StatusCreated, gin.H{
		"status":   "success",
		"message":  "分类创建成功",
		"category": category,
	})
}

// 更新分类
func updateCategory(c *gin.Context) {
	logger.Info("Entered updateCategory handler")

	categoryID, err := strconv.Atoi(c.Param("id"))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "无效的分类ID"})
		return
	}

	var updates map[string]interface{}
	if err := c.ShouldBindJSON(&updates); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "无效的请求数据"})
		return
	}

	var category ForumCategory
	if err := db.First(&category, categoryID).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			c.JSON(http.StatusNotFound, gin.H{"error": "分类不存在"})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取分类失败"})
		return
	}

	if err := db.Model(&category).Updates(updates).Error; err != nil {
		logger.Errorf("Failed to update category: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "更新分类失败"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":   "success",
		"message":  "分类更新成功",
		"category": category,
	})
}

// 删除分类
func deleteCategory(c *gin.Context) {
	logger.Info("Entered deleteCategory handler")

	categoryID, err := strconv.Atoi(c.Param("id"))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "无效的分类ID"})
		return
	}

	if err := db.Delete(&ForumCategory{}, categoryID).Error; err != nil {
		logger.Errorf("Failed to delete category: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "删除分类失败"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":  "success",
		"message": "分类删除成功",
	})
}

// 置顶/取消置顶帖子
func togglePinPost(c *gin.Context) {
	logger.Info("Entered togglePinPost handler")

	postID, err := strconv.Atoi(c.Param("id"))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "无效的帖子ID"})
		return
	}

	var post ForumPost
	if err := db.First(&post, postID).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			c.JSON(http.StatusNotFound, gin.H{"error": "帖子不存在"})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取帖子失败"})
		return
	}

	newPinnedState := !post.IsPinned
	if err := db.Model(&post).Update("is_pinned", newPinnedState).Error; err != nil {
		logger.Errorf("Failed to toggle pin post: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "操作失败"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status": "success",
		"pinned": newPinnedState,
	})
}

// 精选/取消精选帖子
func toggleFeaturePost(c *gin.Context) {
	logger.Info("Entered toggleFeaturePost handler")

	postID, err := strconv.Atoi(c.Param("id"))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "无效的帖子ID"})
		return
	}

	var post ForumPost
	if err := db.First(&post, postID).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			c.JSON(http.StatusNotFound, gin.H{"error": "帖子不存在"})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取帖子失败"})
		return
	}

	newFeaturedState := !post.IsFeatured
	if err := db.Model(&post).Update("is_featured", newFeaturedState).Error; err != nil {
		logger.Errorf("Failed to toggle feature post: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "操作失败"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":   "success",
		"featured": newFeaturedState,
	})
}

type RegisterRequest struct {
	Username string `json:"username" binding:"required"` // 用户名
	Email    string `json:"email" binding:"required"`    // 邮箱
	Password string `json:"password" binding:"required"` // 原始密码
	// 可选：如果前端有电话、公司等字段，可在此添加（与NovaUser对应）
	Telephone string `json:"telephone"`
	Company   string `json:"company"`
}

// RegisterResponse 注册响应结构体
type RegisterResponse struct {
	Code    int    `json:"code"`    // 状态码（200=成功，4xx=失败）
	Message string `json:"message"` // 提示信息
	Data    struct {
		UserID   uint   `json:"user_id"`  // 新增用户ID
		Username string `json:"username"` // 用户名
		Email    string `json:"email"`    // 邮箱
	} `json:"data,omitempty"` // 成功时返回用户信息
}

// RegisterHandler Gin注册接口处理器
// 路由路径：POST /auth/register（解决URL拼写错误问题）
func RegisterHandler(c *gin.Context) {
	// 1. 绑定并验证前端请求参数
	var req RegisterRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, RegisterResponse{
			Code:    400,
			Message: "请求参数错误：" + err.Error(),
		})
		return
	}
	fmt.Print("req:%s", req)

	// 2. 业务逻辑验证（更详细的参数校验）
	if !ValidateUsername(req.Username) {
		c.JSON(http.StatusBadRequest, RegisterResponse{
			Code:    400,
			Message: "用户名格式错误：3-20位，仅支持字母、数字、下划线",
		})
		return
	}
	if !ValidateEmail(req.Email) {
		c.JSON(http.StatusBadRequest, RegisterResponse{
			Code:    400,
			Message: "邮箱格式错误（示例：xxx@xxx.com）",
		})
		return
	}
	if !ValidatePassword(req.Password) {
		c.JSON(http.StatusBadRequest, RegisterResponse{
			Code:    400,
			Message: "密码长度不能少于6位",
		})
		return
	}

	// 3. 检查用户名/邮箱是否已存在（避免重复注册）
	var existingUser NovaUser

	err := db.Where("username = ?", req.Username).
		Or("email = ?", req.Email).
		First(&existingUser).Error

	if err != nil && err != gorm.ErrRecordNotFound {
		// 数据库查询错误
		c.JSON(http.StatusInternalServerError, RegisterResponse{
			Code:    500,
			Message: "服务器错误：查询用户信息失败",
		})
		return
	}
	if err == nil {
		// 用户名或邮箱已存在
		if existingUser.Username == req.Username {
			c.JSON(http.StatusConflict, RegisterResponse{
				Code:    409,
				Message: "用户名已被注册，请更换",
			})
		} else {
			c.JSON(http.StatusConflict, RegisterResponse{
				Code:    409,
				Message: "邮箱已被注册，请更换",
			})
		}
		return
	}

	// 4. 密码加密（使用utils工具函数）

	// 5. 构造NovaUser对象（与数据库表结构对应）
	newUser := NovaUser{
		Username:    req.Username,
		Email:       req.Email,
		Telephone:   req.Telephone,
		Company:     req.Company,
		Role:        "user",                        // 默认普通用户角色（与结构体默认值一致）
		LoginMethod: "local",                       // 默认本地登录（与结构体默认值一致）
		Password:    EncryptedString(req.Password), // 加密后的密码
		GroupName:   "guess",                       // 默认游客组（与结构体默认值一致）
		Profile:     "",                            // 可选：用户简介，默认空
		Extra:       "",                            // 可选：额外信息，默认空
		Activation:  "active",                      // 可选：默认激活（根据业务需求调整）
		Expire:      "",                            // 可选：过期时间，默认空（永久有效）
		Checksum:    "",                            // 可选：校验和，根据业务需求生成
		Timestamp:   GetCurrentTimestamp(),         // 当前时间
	}

	// 6. 插入数据库
	if err := db.Create(&newUser).Error; err != nil {
		c.JSON(http.StatusInternalServerError, RegisterResponse{
			Code:    500,
			Message: "服务器错误：创建用户失败",
		})
		return
	}

	// 7. 注册成功，返回响应
	c.JSON(http.StatusOK, RegisterResponse{
		Code:    200,
		Message: "注册成功！",
		Data: struct {
			UserID   uint   `json:"user_id"`
			Username string `json:"username"`
			Email    string `json:"email"`
		}{
			UserID:   newUser.ID,
			Username: newUser.Username,
			Email:    newUser.Email,
		},
	})
}

// EncryptedString 密码加密类型（与NovaUser结构体匹配）

// HashPassword 密码加密（返回加密后的字符串）
func HashPassword(rawPassword string) (EncryptedString, error) {
	hashedBytes, err := bcrypt.GenerateFromPassword([]byte(rawPassword), bcrypt.DefaultCost)
	if err != nil {
		return "", err
	}
	return EncryptedString(hashedBytes), nil
}

// ValidateEmail 验证邮箱格式
func ValidateEmail(email string) bool {
	emailRegex := regexp.MustCompile(`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`)
	return emailRegex.MatchString(email)
}

// ValidateUsername 验证用户名（3-20位，支持字母、数字、下划线）
func ValidateUsername(username string) bool {
	usernameRegex := regexp.MustCompile(`^[a-zA-Z0-9_]{3,20}$`)
	return usernameRegex.MatchString(username)
}

// ValidatePassword 验证密码（至少6位）
func ValidatePassword(password string) bool {
	return len(password) >= 6
}

// GetCurrentTimestamp 获取当前时间（与NovaUser的Timestamp字段匹配）
func GetCurrentTimestamp() time.Time {
	return time.Now()
}

type ForumAttachment struct {
    ID        uint      `json:"id" gorm:"primaryKey"`
    PostID    uint      `json:"post_id" gorm:"not null"`
    Filename  string    `json:"filename" gorm:"size:255;not null"`
    FilePath  string    `json:"file_path" gorm:"size:500;not null"`
    FileSize  int64     `json:"file_size"`
    FileType  string    `json:"file_type" gorm:"size:100"`
    MimeType  string    `json:"mime_type" gorm:"size:100"`
    IsImage   bool      `json:"is_image" gorm:"default:false"`
    CreatedAt time.Time `json:"created_at" gorm:"autoCreateTime"`
}

// 白名单认证函数
func isUserAuthorized(username string) bool {
    whitelistFile := "authorized_users.txt"

    // 如果白名单文件不存在，默认所有登录用户都可以发帖
    if _, err := os.Stat(whitelistFile); os.IsNotExist(err) {
        return true
    }

    // 读取白名单文件
    content, err := os.ReadFile(whitelistFile)
    if err != nil {
        log.Printf("读取白名单文件失败: %v", err)
        return false
    }

    // 检查用户名是否在白名单中
    lines := strings.Split(string(content), "\n")
    for _, line := range lines {
        if strings.TrimSpace(line) == username {
            return true
        }
    }

    return false
}

func uploadFile(c *gin.Context) {
    logger.Info("Entered uploadFile handler")

    // 获取当前用户并检查认证
    user, ok := getCurrentUser(c)
    if !ok {
        c.JSON(http.StatusUnauthorized, gin.H{"error": "未登录"})
        return
    }

    // 检查用户是否在白名单中
    if !isUserAuthorized(user.Username) {
        c.JSON(http.StatusForbidden, gin.H{"error": "用户未认证，无法上传文件"})
        return
    }

    // 解析表单数据
    postIDStr := c.PostForm("post_id")
    var postID uint
    if postIDStr != "" {
        id, err := strconv.Atoi(postIDStr)
        if err != nil {
            c.JSON(http.StatusBadRequest, gin.H{"error": "无效的帖子ID"})
            return
        }
        postID = uint(id)
    }

    // 处理文件上传
    file, err := c.FormFile("file")
    if err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "文件上传失败: " + err.Error()})
        return
    }

    // 创建上传目录
    uploadDir := "uploads/forum"
    if err := os.MkdirAll(uploadDir, 0755); err != nil {
        logger.Errorf("创建上传目录失败: %v", err)
        c.JSON(http.StatusInternalServerError, gin.H{"error": "服务器错误"})
        return
    }

    // 生成唯一文件名
    fileExt := filepath.Ext(file.Filename)
    fileName := fmt.Sprintf("%d_%s%s", time.Now().UnixNano(), generateRandomString(8), fileExt)
    filePath := filepath.Join(uploadDir, fileName)

    // 保存文件
    if err := c.SaveUploadedFile(file, filePath); err != nil {
        logger.Errorf("保存文件失败: %v", err)
        c.JSON(http.StatusInternalServerError, gin.H{"error": "文件保存失败"})
        return
    }

    // 判断文件类型
    isImage := false
    mimeType := file.Header.Get("Content-Type")
    if strings.HasPrefix(mimeType, "image/") {
        isImage = true
    }

    // 保存到数据库
    attachment := ForumAttachment{
        PostID:   postID,
        Filename: file.Filename,
        FilePath: filePath,
        FileSize: file.Size,
        FileType: fileExt,
        MimeType: mimeType,
        IsImage:  isImage,
    }

    if err := db.Create(&attachment).Error; err != nil {
        logger.Errorf("保存附件信息失败: %v", err)
        // 删除已上传的文件
        os.Remove(filePath)
        c.JSON(http.StatusInternalServerError, gin.H{"error": "保存文件信息失败"})
        return
    }

    c.JSON(http.StatusOK, gin.H{
        "status": "success",
        "data":   attachment,
    })
}

// 生成随机字符串
func generateRandomString(length int) string {
    const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    b := make([]byte, length)
    for i := range b {
        b[i] = charset[rand.Intn(len(charset))]
    }
    return string(b)
}

// 获取帖子附件
func getPostAttachments(c *gin.Context) {
    postID, err := strconv.Atoi(c.Param("id"))
    if err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "无效的帖子ID"})
        return
    }

    var attachments []ForumAttachment
    if err := db.Where("post_id = ?", postID).Find(&attachments).Error; err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": "获取附件失败"})
        return
    }

    c.JSON(http.StatusOK, attachments)
}

// 删除附件
func deleteAttachment(c *gin.Context) {
    attachmentID, err := strconv.Atoi(c.Param("id"))
    if err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "无效的附件ID"})
        return
    }

    var attachment ForumAttachment
    if err := db.First(&attachment, attachmentID).Error; err != nil {
        c.JSON(http.StatusNotFound, gin.H{"error": "附件不存在"})
        return
    }

    // 删除文件
    if err := os.Remove(attachment.FilePath); err != nil {
        logger.Warnf("删除文件失败: %v", err)
    }

    // 删除数据库记录
    if err := db.Delete(&attachment).Error; err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": "删除附件记录失败"})
        return
    }

    c.JSON(http.StatusOK, gin.H{"status": "success", "message": "附件删除成功"})
}
