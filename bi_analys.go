package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"
)

// 注册BI分析相关路由
func RegisterBIRoutes(r *gin.Engine) {
	// BI分析专用接口
	// r.POST("/analysis/bi/pivot", handlePivotAnalysis)         // 透视分析
	r.POST("/analysis/bi/spc", handleSPCAnalysis) // SPC分析
	//r.POST("/analysis/bi/drilldown", handleDrillDown)
	r.POST("/api/bi/drilldown", handleDrillDown)
}

// 读取CSV文件并返回数据和表头
func readCSVFile(filePath string) ([][]string, []string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	headers, err := reader.Read()
	if err != nil {
		return nil, nil, err
	}

	var records [][]string
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, err
		}
		records = append(records, record)
	}

	return records, headers, nil
}

type FilterCondition struct {
	Column string `json:"column"`
	Type   string `json:"type"` // equals, not_equals, greater_than, less_than, between, contains
	Value  string `json:"value"`
	Value2 string `json:"value2,omitempty"` // Only used for "between" type
}

// SPC分析请求结构
type SPCRequest struct {
	FileName       string            `json:"file_name"`
	VariableColumn string            `json:"variable_column"`
	Columns        []string          `json:"columns"` // 列分组列
	SubgroupSize   int               `json:"subgroup_size"`
	ChartType      string            `json:"chart_type"` // xmr, xbar_r, xbar_s
	Filters        []FilterCondition `json:"filters"`    // 支持筛选数据
}

// 过程能力分析结果
type ProcessCapability struct {
	Cp  float64 `json:"cp"`  // 过程能力指数
	Cpk float64 `json:"cpk"` // 过程性能指数
	Pp  float64 `json:"pp"`  // 潜在过程能力指数
	Ppk float64 `json:"ppk"` // 过程性能指数
}

// SPC分析响应结构
type SPCResponse struct {
	Success      bool                 `json:"success"`
	Error        string               `json:"error,omitempty"`
	Mean         float64              `json:"mean"`                 // 总体均值
	StdDev       float64              `json:"std_dev"`              // 总体标准差
	UCL          float64              `json:"ucl"`                  // 上控制限
	LCL          float64              `json:"lcl"`                  // 下控制限
	Data         []map[string]float64 `json:"data"`                 // 带控制限的数据点
	Capability   *ProcessCapability   `json:"capability,omitempty"` // 过程能力分析结果
	TotalPoints  int                  `json:"total_points"`         // 总数据点数
	LoadedPoints int                  `json:"loaded_points"`        // 当前已加载点数
	IsFinished   bool                 `json:"is_finished"`          // 是否加载完成
}

// 处理SPC统计分析（支持分片加载）
func handleSPCAnalysis(c *gin.Context) {
	var req SPCRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": "无效的请求参数: " + err.Error()})
		return
	}

	// 获取分片索引参数（默认从0开始）
	chunkIndex, err := strconv.Atoi(c.DefaultQuery("chunk_index", "0"))
	if err != nil || chunkIndex < 0 {
		c.JSON(400, gin.H{"error": "无效的分片索引"})
		return
	}

	// 设置默认值
	if req.SubgroupSize <= 0 {
		req.SubgroupSize = 5
	}
	if req.ChartType == "" {
		req.ChartType = "xmr" // 默认单值-移动极差图
	}

	// 读取文件
	filePath := filepath.Join("data", req.FileName)
	records, headers, err := readCSVFile(filePath)
	if err != nil {
		c.JSON(500, gin.H{"error": "读取文件失败: " + err.Error()})
		return
	}

	// 应用筛选条件
	filteredRecords := applyAdvancedFilters(records, headers, req.Filters, req.Columns)

	// 查找变量列索引
	varIndex := -1
	for i, header := range headers {
		if header == req.VariableColumn {
			varIndex = i
			break
		}
	}
	if varIndex == -1 {
		c.JSON(400, gin.H{"error": "指定的变量列不存在"})
		return
	}

	// 提取变量值
	var values []float64
	for _, record := range filteredRecords {
		if len(record) <= varIndex {
			continue
		}
		val, err := strconv.ParseFloat(record[varIndex], 64)
		if err == nil {
			values = append(values, val)
		}
	}
	if len(values) == 0 {
		c.JSON(400, gin.H{"error": "没有有效的数值数据"})
		return
	}

	// ########### 新增：大数据抽样处理 ###########
	// 当数据量超过5000时进行抽样，保留最多3000个点（可根据需求调整）
	const maxSamplePoints = 3000
	if len(values) > maxSamplePoints {
		values = sampleData(values, maxSamplePoints)
	}

	// 计算全量（或抽样后）数据的统计量
	var mean, stdDev, ucl, lcl float64
	var allDataPoints []map[string]float64
	var capability *ProcessCapability

	switch req.ChartType {
	case "xbar_r", "xbar_s":
		subgroups := createSubgroups(values, req.SubgroupSize)
		mean, stdDev, ucl, lcl = calculateXBarControlLimits(subgroups, req.ChartType)
		allDataPoints = createXBarDataPoints(subgroups, mean, ucl, lcl, req.ChartType)
		capability = calculateProcessCapability(values, ucl, lcl)
	default:
		mean, stdDev, ucl, lcl = calculateXMRControlLimits(values)
		allDataPoints = createXMRDataPoints(values, mean, ucl, lcl)
		capability = calculateProcessCapability(values, ucl, lcl)
	}

	// ########### 分片处理逻辑 ###########
	const chunkSize = 200 // 每片200个数据点
	totalPoints := len(allDataPoints)
	start := chunkIndex * chunkSize
	end := start + chunkSize

	// 确保end不超过总长度
	if end > totalPoints {
		end = totalPoints
	}

	// 截取当前分片数据
	chunkData := allDataPoints[start:end]

	// 返回分片结果和进度信息
	c.JSON(200, SPCResponse{
		Success:      true,
		Mean:         mean,
		StdDev:       stdDev,
		UCL:          ucl,
		LCL:          lcl,
		Data:         chunkData,
		Capability:   capability,
		TotalPoints:  totalPoints,
		LoadedPoints: end,
		IsFinished:   end >= totalPoints,
	})
}

// 数据抽样函数（等间隔抽样，保留数据分布特性）
func sampleData(data []float64, maxPoints int) []float64 {
	if len(data) <= maxPoints {
		return data
	}
	sampled := make([]float64, 0, maxPoints)
	step := float64(len(data)) / float64(maxPoints)
	for i := 0; i < maxPoints; i++ {
		idx := int(math.Round(float64(i) * step))
		if idx < len(data) {
			sampled = append(sampled, data[idx])
		}
	}
	return sampled
}

// 创建子组（用于Xbar-R和Xbar-S图）
func createSubgroups(data []float64, subgroupSize int) [][]float64 {
	var subgroups [][]float64

	for i := 0; i < len(data); i += subgroupSize {
		end := i + subgroupSize
		if end > len(data) {
			end = len(data)
		}
		subgroup := data[i:end]
		subgroups = append(subgroups, subgroup)
	}

	return subgroups
}

func applyAdvancedFilters(records [][]string, headers []string, filters []FilterCondition, columns []string) [][]string {
	if len(filters) == 0 {
		return records
	}
	filtered := [][]string{}
	for _, record := range records {
		match := true
		for _, filter := range filters {
			// 查找列索引
			colIndex := -1
			for i, header := range headers {
				if header == filter.Column {
					colIndex = i
					break
				}
			}
			if colIndex == -1 || colIndex >= len(record) {
				match = false
				break
			}
			recordValue := record[colIndex]
			// 尝试转换为数值（允许空值或非数值通过基础筛选）
			numVal, numErr := strconv.ParseFloat(recordValue, 64)
			filterNum, filterErr := strconv.ParseFloat(filter.Value, 64)
			// 根据筛选类型处理
			switch filter.Type {
			case "equals":
				if numErr == nil && filterErr == nil {
					if numVal != filterNum {
						match = false
					}
				} else {
					// 字符串比较（不区分大小写）
					if strings.ToLower(recordValue) != strings.ToLower(filter.Value) {
						match = false
					}
				}
			case "contains":
				if !strings.Contains(strings.ToLower(recordValue), strings.ToLower(filter.Value)) {
					match = false
				}
			// 其他筛选类型（greater_than/between等）仅对数值生效
			case "greater_than", "less_than", "between":
				if numErr != nil || filterErr != nil {
					match = false // 非数值不满足数值筛选条件
				} else {
					if filter.Type == "greater_than" && numVal <= filterNum {
						match = false
					} else if filter.Type == "less_than" && numVal >= filterNum {
						match = false
					} else if filter.Type == "between" {
						filterNum2, err := strconv.ParseFloat(filter.Value2, 64)
						if err != nil || numVal < math.Min(filterNum, filterNum2) || numVal > math.Max(filterNum, filterNum2) {
							match = false
						}
					}
				}
			}
			if !match {
				break
			}
		}
		if match {
			filtered = append(filtered, record)
		}
	}
	return filtered
}

func stringContains(str, substr string) bool {
	if len(substr) == 0 {
		return true // 空字符串匹配所有
	}
	for i := 0; i <= len(str)-len(substr); i++ {
		if str[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// 计算Xbar-R/Xbar-S控制限
func calculateXBarControlLimits(subgroups [][]float64, chartType string) (mean, stdDev, ucl, lcl float64) {
	// 计算每个子组的均值
	subgroupMeans := make([]float64, len(subgroups))
	for i, subgroup := range subgroups {
		subgroupMeans[i] = calculateMean(subgroup)
	}

	// 计算总均值
	mean = calculateMean(subgroupMeans)

	// 计算极差或标准差
	var rangesOrStds []float64
	if chartType == "xbar_r" {
		// 极差
		for _, subgroup := range subgroups {
			if len(subgroup) < 2 {
				rangesOrStds = append(rangesOrStds, 0)
				continue
			}
			minVal := subgroup[0]
			maxVal := subgroup[0]
			for _, val := range subgroup[1:] {
				if val < minVal {
					minVal = val
				}
				if val > maxVal {
					maxVal = val
				}
			}
			rangesOrStds = append(rangesOrStds, maxVal-minVal)
		}
	} else {
		// 标准差
		for _, subgroup := range subgroups {
			m := calculateMean(subgroup)
			rangesOrStds = append(rangesOrStds, calculateStdDev(subgroup, m))
		}
	}

	// 计算平均极差或平均标准差
	avgRangeOrStd := calculateMean(rangesOrStds)

	// 根据子组大小获取控制限系数
	n := len(subgroups[0])
	var a2 float64

	switch n {
	case 2:
		a2 = 1.880
	case 3:
		a2 = 1.023
	case 4:
		a2 = 0.729
	case 5:
		a2 = 0.577
	case 6:
		a2 = 0.483
	case 7:
		a2 = 0.419
	case 8:
		a2 = 0.373
	case 9:
		a2 = 0.337
	case 10:
		a2 = 0.308
	default:
		a2 = 0.577 // 默认使用子组大小为5的系数
	}

	// 计算控制限
	ucl = mean + a2*avgRangeOrStd
	lcl = mean - a2*avgRangeOrStd

	// 计算总体标准差
	allData := []float64{}
	for _, sg := range subgroups {
		allData = append(allData, sg...)
	}
	stdDev = calculateStdDev(allData, calculateMean(allData))

	return mean, stdDev, ucl, lcl
}

// 创建Xbar-R/Xbar-S图的数据点
func createXBarDataPoints(subgroups [][]float64, mean, ucl, lcl float64, chartType string) []map[string]float64 {
	dataPoints := make([]map[string]float64, len(subgroups))
	// 计算每个子组的极差
	subgroupRanges := make([]float64, len(subgroups))
	for i, subgroup := range subgroups {
		if len(subgroup) < 2 {
			subgroupRanges[i] = 0
		} else {
			minVal := subgroup[0]
			maxVal := subgroup[0]
			for _, val := range subgroup[1:] {
				if val < minVal {
					minVal = val
				}
				if val > maxVal {
					maxVal = val
				}
			}
			subgroupRanges[i] = maxVal - minVal // 子组极差
		}
	}
	// 生成数据点（包含极差）
	for i, subgroup := range subgroups {
		subgroupMean := calculateMean(subgroup)
		point := map[string]float64{
			"index": float64(i),
			"value": subgroupMean,
			"mean":  mean,
			"ucl":   ucl,
			"lcl":   lcl,
			"range": subgroupRanges[i], // 添加极差字段
		}
		dataPoints[i] = point
	}
	return dataPoints
}

// func sampleData(data []float64, maxPoints int) []float64 {
// 	if len(data) <= maxPoints {
// 		return data // 数据量小，全量返回
// 	}
// 	// 等间隔抽样（保证分布均匀）
// 	step := float64(len(data)) / float64(maxPoints)
// 	sampled := make([]float64, 0, maxPoints)
// 	for i := 0; i < maxPoints; i++ {
// 		idx := int(math.Round(float64(i) * step))
// 		if idx < len(data) {
// 			sampled = append(sampled, data[idx])
// 		}
// 	}
// 	return sampled
// }

// 修改：计算XMR控制限时应用抽样
func calculateXMRControlLimits(data []float64) (mean, stdDev, ucl, lcl float64) {
	// 限制SPC数据点最大数量（根据屏幕可展示范围，200-500点足够）
	maxSPPPoints := 1000
	sampledData := sampleData(data, maxSPPPoints)

	mean = calculateMean(sampledData) // 用抽样数据计算均值（更高效）

	// 计算移动极差（基于抽样数据）
	movingRanges := make([]float64, len(sampledData)-1)
	for i := 1; i < len(sampledData); i++ {
		movingRanges[i-1] = math.Abs(sampledData[i] - sampledData[i-1])
	}
	avgMovingRange := calculateMean(movingRanges)

	// 控制限计算（保持原逻辑，基于抽样数据的统计量）
	ucl = mean + 2.66*avgMovingRange
	lcl = mean - 2.66*avgMovingRange
	stdDev = calculateStdDev(sampledData, mean)
	return mean, stdDev, ucl, lcl
}

// 修改：创建XMR数据点时使用抽样后的数据
func createXMRDataPoints(data []float64, mean, ucl, lcl float64) []map[string]float64 {
	maxSPPPoints := 1000
	sampledData := sampleData(data, maxSPPPoints)

	dataPoints := make([]map[string]float64, len(sampledData))
	for i, val := range sampledData {
		point := map[string]float64{
			"index": float64(i),
			"value": val,
			"mean":  mean,
			"ucl":   ucl,
			"lcl":   lcl,
			"range": 0,
		}
		if i > 0 {
			point["range"] = math.Abs(sampledData[i] - sampledData[i-1])
		}
		dataPoints[i] = point
	}
	return dataPoints
}

// 计算X-MR控制限
// func calculateXMRControlLimits(data []float64) (mean, stdDev, ucl, lcl float64) {
// 	mean = calculateMean(data)

// 	// 计算移动极差
// 	movingRanges := make([]float64, len(data)-1)
// 	for i := 1; i < len(data); i++ {
// 		movingRanges[i-1] = math.Abs(data[i] - data[i-1])
// 	}

// 	// 计算平均移动极差
// 	avgMovingRange := calculateMean(movingRanges)

// 	// 计算控制限 (使用D4=3.267系数)
// 	ucl = mean + 2.66*avgMovingRange // 对于X-MR图，使用2.66作为系数
// 	lcl = mean - 2.66*avgMovingRange

// 	// 计算总体标准差
// 	stdDev = calculateStdDev(data, mean)

// 	return mean, stdDev, ucl, lcl
// }

// // 创建X-MR图的数据点
// func createXMRDataPoints(data []float64, mean, ucl, lcl float64) []map[string]float64 {
// 	dataPoints := make([]map[string]float64, len(data))
// 	for i, val := range data {
// 		point := map[string]float64{
// 			"index": float64(i),
// 			"value": val,
// 			"mean":  mean,
// 			"ucl":   ucl,
// 			"lcl":   lcl,
// 			"range": 0, // 初始化极差为0
// 		}
// 		// 计算移动极差（从第二个数据点开始）
// 		if i > 0 {
// 			point["range"] = math.Abs(data[i] - data[i-1])
// 		}
// 		dataPoints[i] = point
// 	}
// 	return dataPoints
// }

// 计算过程能力
func calculateProcessCapability(data []float64, ucl, lcl float64) *ProcessCapability {
	if len(data) < 2 || ucl <= lcl {
		return nil
	}

	mean := calculateMean(data)
	stdDev := calculateStdDev(data, mean)

	// 计算过程能力指数
	cp := (ucl - lcl) / (6 * stdDev)

	// 计算Cpk
	cpu := (ucl - mean) / (3 * stdDev)
	cpl := (mean - lcl) / (3 * stdDev)
	cpk := math.Min(cpu, cpl)

	// 对于Pp和Ppk，这里简化处理，与Cp和Cpk相同
	pp := cp
	ppk := cpk

	return &ProcessCapability{
		Cp:  cp,
		Cpk: cpk,
		Pp:  pp,
		Ppk: ppk,
	}
}

// 辅助函数：计算均值
func calculateMean(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}

	sum := 0.0
	for _, val := range data {
		sum += val
	}

	return sum / float64(len(data))
}

// 辅助函数：计算标准差
func calculateStdDev(data []float64, mean float64) float64 {
	if len(data) < 2 {
		return 0
	}

	sumSq := 0.0
	for _, val := range data {
		diff := val - mean
		sumSq += diff * diff
	}

	variance := sumSq / float64(len(data)-1)
	return math.Sqrt(variance)
}

func applyFilters(records [][]string, headers []string, filters []FilterCondition) [][]string {
	if len(filters) == 0 {
		return records
	}

	// Create header index map
	headerIndex := make(map[string]int)
	for i, h := range headers {
		headerIndex[h] = i
	}

	var filtered [][]string

	for _, record := range records {
		matches := true

		for _, filter := range filters {
			colIndex, exists := headerIndex[filter.Column]
			if !exists || colIndex >= len(record) {
				matches = false
				break
			}

			recordValue := strings.TrimSpace(record[colIndex])
			filterValue := strings.TrimSpace(filter.Value)
			filterValue2 := strings.TrimSpace(filter.Value2)

			// Try numeric comparison first
			recordNum, recordNumErr := strconv.ParseFloat(recordValue, 64)
			filterNum, filterNumErr := strconv.ParseFloat(filterValue, 64)
			filterNum2, filterNum2Err := strconv.ParseFloat(filterValue2, 64)

			switch filter.Type {
			case "equals":
				if recordNumErr == nil && filterNumErr == nil {
					if recordNum != filterNum {
						matches = false
					}
				} else {
					if recordValue != filterValue {
						matches = false
					}
				}
			case "not_equals":
				if recordNumErr == nil && filterNumErr == nil {
					if recordNum == filterNum {
						matches = false
					}
				} else {
					if recordValue == filterValue {
						matches = false
					}
				}
			case "greater_than":
				if recordNumErr == nil && filterNumErr == nil {
					if recordNum <= filterNum {
						matches = false
					}
				} else {
					// Fallback to string comparison
					if recordValue <= filterValue {
						matches = false
					}
				}
			case "less_than":
				if recordNumErr == nil && filterNumErr == nil {
					if recordNum >= filterNum {
						matches = false
					}
				} else {
					if recordValue >= filterValue {
						matches = false
					}
				}
			case "greater_equals":
				if recordNumErr == nil && filterNumErr == nil {
					if recordNum < filterNum {
						matches = false
					}
				} else {
					if recordValue < filterValue {
						matches = false
					}
				}
			case "less_equals":
				if recordNumErr == nil && filterNumErr == nil {
					if recordNum > filterNum {
						matches = false
					}
				} else {
					if recordValue > filterValue {
						matches = false
					}
				}
			case "between":
				if recordNumErr == nil && filterNumErr == nil && filterNum2Err == nil {
					minVal := math.Min(filterNum, filterNum2)
					maxVal := math.Max(filterNum, filterNum2)
					if recordNum < minVal || recordNum > maxVal {
						matches = false
					}
				} else {
					// Fallback to string comparison
					if recordValue < filterValue || recordValue > filterValue2 {
						matches = false
					}
				}
			case "contains":
				if !strings.Contains(strings.ToLower(recordValue), strings.ToLower(filterValue)) {
					matches = false
				}
			default:
				matches = false
			}

			if !matches {
				break
			}
		}

		if matches {
			filtered = append(filtered, record)
		}
	}

	return filtered
}

// ####################################################
// 下钻分析请求结构

type Filter struct {
	Column string `json:"column"`           // 筛选列名
	Type   string `json:"type"`             // 筛选类型：equals, not_equals, greater_than, less_than, between, contains
	Value  string `json:"value"`            // 筛选值1
	Value2 string `json:"value2,omitempty"` // 筛选值2（用于between类型）
}
type DrillStep struct {
	Dimension string `json:"dimension"` // 维度名称（如"区域"）
	Value     string `json:"value"`     // 钻取值（如"华东"）
}

// 下钻分析请求结构体
type DrillDownRequest struct {
	FileName    string      `json:"file_name"`   // 文件名
	Dimension   string      `json:"dimension"`   // 当前分析维度
	Measure     string      `json:"measure"`     // 度量字段
	Aggregation string      `json:"aggregation"` // 聚合方式：count, sum, avg等
	DrillPath   []DrillStep `json:"drill_path"`  // 钻取路径（历史步骤）
	Filters     []Filter    `json:"filters"`     // 筛选条件
	Action      string      `json:"action"`      // 操作类型：init, drill, rollup
	DrillValue  string      `json:"drill_value"` // 钻取值
	ChunkIndex  int         `json:"chunk_index"` // 分片索引（默认0）
}

// 下钻分析响应结构体
type DrillDownResponse struct {
	Success             bool        `json:"success"`
	Error               string      `json:"error,omitempty"`
	Data                []DrillItem `json:"data"`                 // 当前分片数据
	Total               int64       `json:"total"`                // 总度量值（兼容int64）
	TotalGroups         int         `json:"total_groups"`         // 总分组数
	LoadedGroups        int         `json:"loaded_groups"`        // 已加载组数
	IsFinished          bool        `json:"is_finished"`          // 是否加载完成
	CurrentLevel        int         `json:"current_level"`        // 当前钻取层级
	AvailableDimensions []string    `json:"available_dimensions"` // 可下钻维度
}

// 下钻分析结果项
type DrillItem struct {
	DimensionValue string  `json:"dimensionValue"` // 维度值
	MeasureValue   float64 `json:"measureValue"`   // 度量值（聚合结果）
	RawValue       float64 `json:"rawValue"`       // 原始值（可选）
	Count          int     `json:"count"`          // 样本数
}
type DrillPathStep struct {
	Dimension     string      `json:"dimension"`
	Value         interface{} `json:"value"`
	NextDimension string      `json:"next_dimension"`
}

// // 注册下钻分析路由
// func RegisterBIRoutes(r *gin.Engine) {
// 	r.POST("/analysis/bi/drilldown", handleDrillDown)
// }

// 处理下钻分析请求
func handleDrillDown(c *gin.Context) {
	var req DrillDownRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": "无效的请求参数: " + err.Error()})
		return
	}

	// 处理分片索引默认值
	if req.ChunkIndex < 0 {
		req.ChunkIndex = 0
	}

	// 读取文件数据
	filePath := filepath.Join("data", req.FileName)
	records, headers, err := readCSVFile(filePath)
	if err != nil {
		c.JSON(500, gin.H{"error": "读取文件失败: " + err.Error()})
		return
	}

	// 应用筛选条件（修复类型不匹配问题）
	filteredData := applyDrillDownFilters(records, headers, req.Filters, req.DrillPath)

	// 执行下钻分析
	fullResult, availableDims, err := performDrillDownAnalysis(filteredData, headers, req)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	// 分片处理逻辑
	const chunkSize = 50
	totalGroups := len(fullResult)
	start := req.ChunkIndex * chunkSize
	end := start + chunkSize

	// 处理边界情况
	if start >= totalGroups {
		c.JSON(200, DrillDownResponse{
			Success:             true,
			Data:                []DrillItem{},
			Total:               0,
			TotalGroups:         totalGroups,
			LoadedGroups:        totalGroups,
			IsFinished:          true,
			CurrentLevel:        len(req.DrillPath),
			AvailableDimensions: availableDims,
		})
		return
	}
	if end > totalGroups {
		end = totalGroups
	}

	// 截取当前分片数据
	chunkData := fullResult[start:end]

	// 计算总度量值
	total := calculateDrillDownTotal(fullResult)

	// 返回响应
	c.JSON(200, DrillDownResponse{
		Success:             true,
		Data:                chunkData,
		Total:               total,
		TotalGroups:         totalGroups,
		LoadedGroups:        end,
		IsFinished:          end >= totalGroups,
		CurrentLevel:        len(req.DrillPath),
		AvailableDimensions: availableDims,
	})
}

// 应用下钻特定筛选条件
func applyDrillDownFilters(records [][]string, headers []string, filters []Filter, drillPath []DrillStep) [][]string {
	headerIndex := make(map[string]int)
	for i, h := range headers {
		headerIndex[h] = i
	}

	var filtered [][]string

	for _, record := range records {
		match := true

		// 应用常规筛选条件
		for _, filter := range filters {
			colIndex, exists := headerIndex[filter.Column]
			if !exists || colIndex >= len(record) {
				match = false
				break
			}

			if !evaluateFilterCondition(record[colIndex], filter) {
				match = false
				break
			}
		}

		// 应用钻取路径筛选（基于历史步骤）
		for _, step := range drillPath {
			colIndex, exists := headerIndex[step.Dimension]
			if !exists || colIndex >= len(record) {
				match = false
				break
			}

			recordValue := strings.TrimSpace(record[colIndex])
			if recordValue != step.Value {
				match = false
				break
			}
		}

		if match {
			filtered = append(filtered, record)
		}
	}

	return filtered
}

// 新增：合并小占比分组为"其他"
func mergeSmallGroups(groups []DrillItem, minRatio float64) []DrillItem {
	if len(groups) <= 50 { // 分组少，不合并
		return groups
	}
	// 计算总数量
	total := 0
	for _, g := range groups {
		total += g.Count
	}
	if total == 0 {
		return groups
	}

	// 分离大分组和小分组
	var largeGroups []DrillItem
	smallTotal := 0.0
	smallCount := 0
	smallRaw := 0.0

	for _, g := range groups {
		ratio := float64(g.Count) / float64(total)
		if ratio >= minRatio { // 保留占比≥1%的分组
			largeGroups = append(largeGroups, g)
		} else { // 合并小分组
			smallTotal += g.MeasureValue
			smallCount += g.Count
			smallRaw += g.RawValue
		}
	}

	// 添加"其他"分组（如果有小分组）
	if smallCount > 0 {
		largeGroups = append(largeGroups, DrillItem{
			DimensionValue: "其他",
			MeasureValue:   smallTotal,
			RawValue:       smallRaw,
			Count:          smallCount,
		})
	}

	// 限制最大返回50个分组
	if len(largeGroups) > 50 {
		largeGroups = largeGroups[:50]
	}
	return largeGroups
}

func performDrillDownAnalysis(
	filteredData [][]string,
	headers []string,
	req DrillDownRequest,
) ([]DrillItem, []string, error) {
	// ########### 大数据处理：抽样 + 分组合并 ###########
	// 当数据量过大时先抽样（保留10%或最多10万条，取较小值）
	sampledData := filteredData
	if len(filteredData) > 100000 {
		sampledData = sampleRecords(filteredData, 100000) // 抽样函数
	}

	// 执行分组统计（基于抽样后的数据，需传入聚合方式）
	result, err := calculateDrillGroups(
		sampledData,
		headers,
		req.Dimension,
		req.Measure,
		req.Aggregation, // 补充聚合方式参数（原代码遗漏）
	)
	if err != nil {
		return nil, nil, fmt.Errorf("分组计算失败: %w", err)
	}

	// 排序并合并小分组（保持与前端展示逻辑一致）
	sort.Slice(result, func(i, j int) bool {
		return result[i].MeasureValue > result[j].MeasureValue
	})
	mergedResult := mergeSmallGroups(result, 0.01) // 合并占比<1%的组

	// 确定可用下钻维度（需考虑当前钻取路径）
	availableDims := findAvailableDimensions(headers, req.Dimension, req.DrillPath)

	return mergedResult, availableDims, nil
}

// 记录抽样函数（优化：按比例保留数据分布，避免偏移导致的抽样偏差）
func sampleRecords(records [][]string, maxCount int) [][]string {
	if len(records) <= maxCount {
		return records
	}
	sampled := make([][]string, 0, maxCount)
	step := float64(len(records)) / float64(maxCount)

	// 改进：使用固定种子偏移，确保同一份数据抽样结果一致（便于调试）
	offset := int(step) / 2 // 取步长的一半作为固定偏移，避免随机波动

	for i := 0; i < maxCount; i++ {
		idx := int(float64(i)*step) + offset
		if idx < len(records) {
			sampled = append(sampled, records[idx])
		}
	}
	return sampled
}

// 下钻分组结构
type drillDownGroup struct {
	dimensionValue string
	count          int
	sum            float64
	min            float64
	max            float64
	values         []float64
}

// 计算下钻结果总数
func calculateDrillDownTotal(items []DrillItem) int64 {
	var total int64
	for _, item := range items {
		total += int64(item.Count)
	}
	return total
}

// 查找可用的下钻维度
func findAvailableDimensions(headers []string, currentDim string, drillPath []DrillStep) []string {
	// 排除已使用的维度
	usedDims := make(map[string]bool)
	usedDims[currentDim] = true
	for _, step := range drillPath {
		usedDims[step.Dimension] = true
	}

	var available []string
	for _, h := range headers {
		if !usedDims[h] && h != "" {
			available = append(available, h)
		}
	}
	return available
}

// 评估筛选条件
func evaluateFilterCondition(recordValue string, filter Filter) bool {
	recordValue = strings.TrimSpace(recordValue)
	filterValue := strings.TrimSpace(filter.Value)
	filterValue2 := strings.TrimSpace(filter.Value2)

	// 尝试数值比较
	recordNum, recordNumErr := strconv.ParseFloat(recordValue, 64)
	filterNum, filterNumErr := strconv.ParseFloat(filterValue, 64)
	filterNum2, filterNum2Err := strconv.ParseFloat(filterValue2, 64)

	switch filter.Type {
	case "equals":
		if recordNumErr == nil && filterNumErr == nil {
			return recordNum == filterNum
		}
		return strings.EqualFold(recordValue, filterValue)
	case "not_equals":
		if recordNumErr == nil && filterNumErr == nil {
			return recordNum != filterNum
		}
		return !strings.EqualFold(recordValue, filterValue)
	case "greater_than":
		if recordNumErr == nil && filterNumErr == nil {
			return recordNum > filterNum
		}
		return recordValue > filterValue
	case "less_than":
		if recordNumErr == nil && filterNumErr == nil {
			return recordNum < filterNum
		}
		return recordValue < filterValue
	case "greater_equals":
		if recordNumErr == nil && filterNumErr == nil {
			return recordNum >= filterNum
		}
		return recordValue >= filterValue
	case "less_equals":
		if recordNumErr == nil && filterNumErr == nil {
			return recordNum <= filterNum
		}
		return recordValue <= filterValue
	case "between":
		if recordNumErr == nil && filterNumErr == nil && filterNum2Err == nil {
			minVal := math.Min(filterNum, filterNum2)
			maxVal := math.Max(filterNum, filterNum2)
			return recordNum >= minVal && recordNum <= maxVal
		}
		return recordValue >= filterValue && recordValue <= filterValue2
	case "contains":
		return strings.Contains(strings.ToLower(recordValue), strings.ToLower(filterValue))
	default:
		return false
	}
}

func calculateDrillGroups(
	data [][]string,
	headers []string,
	dimension string,
	measure string,
	aggregation string,
) ([]DrillItem, error) {
	// 查找维度列和度量列索引
	dimIndex := -1
	measureIndex := -1
	for i, h := range headers {
		if h == dimension {
			dimIndex = i
		}
		if h == measure {
			measureIndex = i
		}
	}
	if dimIndex == -1 {
		return nil, fmt.Errorf("维度字段不存在: %s", dimension)
	}

	// 按维度分组统计
	groups := make(map[string]*DrillItem)
	for _, record := range data {
		if dimIndex >= len(record) {
			continue // 跳过不完整记录
		}
		dimVal := strings.TrimSpace(record[dimIndex])
		if dimVal == "" {
			continue // 跳过空维度值
		}

		// 初始化分组
		if _, exists := groups[dimVal]; !exists {
			groups[dimVal] = &DrillItem{
				DimensionValue: dimVal,
				MeasureValue:   0,
				RawValue:       0,
				Count:          0,
			}
		}
		group := groups[dimVal]
		group.Count++ // 计数默认+1

		// 处理度量字段（如果指定）
		if measure != "" && measureIndex != -1 && measureIndex < len(record) {
			measureStr := strings.TrimSpace(record[measureIndex])
			measureVal, err := strconv.ParseFloat(measureStr, 64)
			if err != nil {
				continue // 跳过无效度量值
			}

			// 根据聚合方式计算
			switch aggregation {
			case "sum":
				group.MeasureValue += measureVal
			case "avg":
				group.MeasureValue = (group.MeasureValue*float64(group.Count-1) + measureVal) / float64(group.Count)
			case "min":
				if group.Count == 1 || measureVal < group.MeasureValue {
					group.MeasureValue = measureVal
				}
			case "max":
				if group.Count == 1 || measureVal > group.MeasureValue {
					group.MeasureValue = measureVal
				}
			case "count":
				// 计数已在上面处理
			default:
				group.MeasureValue += measureVal // 默认求和
			}
			group.RawValue += measureVal // 累计原始值（用于后续计算）
		}
	}

	// 转换为切片
	result := make([]DrillItem, 0, len(groups))
	for _, item := range groups {
		result = append(result, *item)
	}
	return result, nil
}
