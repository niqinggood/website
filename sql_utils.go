package main

import (
	"regexp"
	"strings"
)

// 定义风险等级
const (
	RiskLevelSafe    = "safe"    // 安全
	RiskLevelWarning = "warning" // 警告
	RiskLevelDanger  = "danger"  // 危险
)

// SqlRisk 包含SQL风险检测结果
type SqlRisk struct {
	IsRisky   bool   // 是否有风险
	RiskLevel string // 风险等级
	Reason    string // 风险原因
}

// CheckSqlRisk 检查SQL语句是否包含风险操作
func CheckSqlRisk(sql string) SqlRisk {
	// 预处理SQL：转为大写，便于统一匹配；去除注释，避免干扰检测
	processedSql := preprocessSql(sql)

	// 定义危险操作模式 - 会直接修改/删除表结构或大量数据
	dangerPatterns := []*regexp.Regexp{
		// DROP相关操作
		regexp.MustCompile(`\bDROP\s+(TABLE|DATABASE|SCHEMA|VIEW|INDEX)\b`),
		// ALTER修改表结构
		regexp.MustCompile(`\bALTER\s+TABLE\b`),
		// TRUNCATE清空表
		regexp.MustCompile(`\bTRUNCATE\s+(TABLE)?\b`),
		// DELETE不带WHERE条件
		regexp.MustCompile(`\bDELETE\s+FROM\s+\w+\s*;`),
		regexp.MustCompile(`\bDELETE\s+FROM\s+\w+\s*$`),
	}

	// 检查危险操作
	for _, pattern := range dangerPatterns {
		if pattern.MatchString(processedSql) {
			return SqlRisk{
				IsRisky:   true,
				RiskLevel: RiskLevelDanger,
				Reason:    "检测到危险操作: " + extractMatchedText(pattern, processedSql),
			}
		}
	}

	// 定义警告操作模式 - 可能修改数据但相对可控
	warningPatterns := []*regexp.Regexp{
		// UPDATE操作
		regexp.MustCompile(`\bUPDATE\s+\w+\b`),
		// DELETE带WHERE条件
		regexp.MustCompile(`\bDELETE\s+FROM\s+\w+\s+WHERE\b`),
		// INSERT操作
		regexp.MustCompile(`\bINSERT\s+(INTO)?\s+\w+\b`),
		// CREATE/DROP临时表
		regexp.MustCompile(`\bCREATE\s+TEMPORARY\s+TABLE\b`),
		regexp.MustCompile(`\bDROP\s+TEMPORARY\s+TABLE\b`),
	}

	// 检查警告操作
	for _, pattern := range warningPatterns {
		if pattern.MatchString(processedSql) {
			return SqlRisk{
				IsRisky:   true,
				RiskLevel: RiskLevelWarning,
				Reason:    "检测到需要注意的操作: " + extractMatchedText(pattern, processedSql),
			}
		}
	}

	// 检查是否为SELECT语句（最安全的操作）
	selectPattern := regexp.MustCompile(`^\s*\bSELECT\b`)
	if selectPattern.MatchString(processedSql) {
		return SqlRisk{
			IsRisky:   false,
			RiskLevel: RiskLevelSafe,
			Reason:    "安全的查询操作",
		}
	}

	// 其他未识别的SQL操作
	return SqlRisk{
		IsRisky:   true,
		RiskLevel: RiskLevelWarning,
		Reason:    "检测到未分类的SQL操作，建议审核",
	}
}

// 预处理SQL：转为大写，去除注释
func preprocessSql(sql string) string {
	// 转为大写
	sqlUpper := strings.ToUpper(sql)

	// 去除单行注释 --
	sqlWithoutLineComments := regexp.MustCompile(`--.*$`).ReplaceAllString(sqlUpper, " ")

	// 去除多行注释 /* */
	sqlWithoutComments := regexp.MustCompile(`/\*.*?\*/`).ReplaceAllString(sqlWithoutLineComments, " ")

	// 替换多个空格为单个空格
	return regexp.MustCompile(`\s+`).ReplaceAllString(sqlWithoutComments, " ")
}

// 提取匹配到的文本（简化版）
func extractMatchedText(pattern *regexp.Regexp, sql string) string {
	match := pattern.FindString(sql)
	if len(match) > 50 {
		return match[:50] + "..."
	}
	return match
}
