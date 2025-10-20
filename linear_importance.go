package main

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"math"
	"log"
	"strconv"
	"math/rand"
	"time"
)

// 定义任务类型（分类/回归）
type TaskType string

// const (
// 	TaskClassification TaskType = "classification"
// 	TaskRegression     TaskType = "regression"
// )
//
// // FeatureImportance 特征重要性结果结构
// type FeatureImportance struct {
// 	Feature    string  `json:"feature"`
// 	Importance float64 `json:"importance"`
// 	Type       string  `json:"type"`
// }

// 主函数：计算特征重要性
// func calculateFeatureImportance(headers []string, records [][]string, target string, features []string, algorithms []string) []FeatureImportance {
// 	// 1. 定位目标列和特征列
// 	targetIdx := -1
// 	for i, h := range headers {
// 		if h == target {
// 			targetIdx = i
// 			break
// 		}
// 	}
// 	if targetIdx == -1 {
// 		log.Printf("目标列 %s 不存在", target)
// 		return nil
// 	}
//
// 	// 2. 收集特征列索引
// 	var featureIndices []int
// 	var featureNames []string
// 	for _, f := range features {
// 		for i, h := range headers {
// 			if h == f && i != targetIdx {
// 				featureIndices = append(featureIndices, i)
// 				featureNames = append(featureNames, f)
// 				break
// 			}
// 		}
// 	}
// 	if len(featureIndices) == 0 {
// 		log.Printf("未找到有效特征列")
// 		return nil
// 	}
//
// 	// 3. 转换为数值矩阵（假设已实现convertToMatrix）
// 	X, y, taskType := convertToMatrix(headers, records, featureIndices, targetIdx)
// 	if X == nil || y == nil {
// 		log.Printf("矩阵转换失败")
// 		return nil
// 	}
//
// 	// 4. 根据任务类型选择算法
// 	var allImportances []FeatureImportance
// 	for _, algo := range algorithms {
// 		switch algo {
// 		case "rf":
// 			if taskType == TaskClassification {
// 				allImportances = append(allImportances, randomForestClassificationImportance(X, y, featureNames)...)
// 			} else {
// 				allImportances = append(allImportances, randomForestRegressionImportance(X, y, featureNames)...)
// 			}
// 		case "linear":
// 			if taskType == TaskClassification {
// 				allImportances = append(allImportances, logisticRegressionImportance(X, y, featureNames)...)
// 			} else {
// 				allImportances = append(allImportances, linearRegressionImportance(X, y, featureNames)...)
// 			}
// 		case "mi":
// 			allImportances = append(allImportances, mutualInformationImportance(X, y, featureNames, taskType)...)
// 		}
// 	}
//
// 	// 5. 标准化每个算法的重要性分数
// 	algorithmGroups := make(map[string][]FeatureImportance)
// 	for _, imp := range allImportances {
// 		algorithmGroups[imp.Type] = append(algorithmGroups[imp.Type], imp)
// 	}
//
// 	normalized := make([]FeatureImportance, 0, len(allImportances))
// 	for _, group := range algorithmGroups {
// 		maxVal := 0.0
// 		for _, imp := range group {
// 			if imp.Importance > maxVal {
// 				maxVal = imp.Importance
// 			}
// 		}
// 		for _, imp := range group {
// 			normImp := imp
// 			if maxVal > 0 {
// 				normImp.Importance = imp.Importance / maxVal
// 			}
// 			normalized = append(normalized, normImp)
// 		}
// 	}
//
// 	return normalized
// }

// // 随机森林分类特征重要性
// func randomForestClassificationImportance(X, y mat.Matrix, featureNames []string) []FeatureImportance {
// 	xDense := ensureDenseMatrix(X)
// 	yDense := ensureDenseMatrix(y)
// 	if xDense == nil || yDense == nil {
// 		log.Printf("随机森林分类：矩阵转换失败")
// 		return nil
// 	}
//
// 	rows, cols := xDense.Dims()
// 	importances := make([]FeatureImportance, cols)
// 	// 简化实现：实际应使用完整的随机森林逻辑
// 	for i, name := range featureNames {
// 		importances[i] = FeatureImportance{
// 			Feature:    name,
// 			Importance: math.Abs(rand.NormFloat64() * 0.5), // 模拟随机森林重要性
// 			Type:       "rf_classification",
// 		}
// 	}
// 	return importances
// }
//
// // 随机森林回归特征重要性
// func randomForestRegressionImportance(X, y mat.Matrix, featureNames []string) []FeatureImportance {
// 	xDense := ensureDenseMatrix(X)
// 	yDense := ensureDenseMatrix(y)
// 	if xDense == nil || yDense == nil {
// 		log.Printf("随机森林回归：矩阵转换失败")
// 		return nil
// 	}
//
// 	rows, cols := xDense.Dims()
// 	importances := make([]FeatureImportance, cols)
// 	// 简化实现
// 	for i, name := range featureNames {
// 		importances[i] = FeatureImportance{
// 			Feature:    name,
// 			Importance: math.Abs(rand.NormFloat64() * 0.5), // 模拟随机森林重要性
// 			Type:       "rf_regression",
// 		}
// 	}
// 	return importances
// }

// 逻辑回归特征重要性（分类问题）
func logisticRegressionImportance(X, y mat.Matrix, featureNames []string) []FeatureImportance {
	log.Printf("开始逻辑回归特征重要性计算")
	xDense := ensureDenseMatrix(X)
	yDense := ensureDenseMatrix(y)
	if xDense == nil || yDense == nil {
		log.Printf("逻辑回归：矩阵转换失败")
		return nil
	}

	rows, cols := xDense.Dims()

	// 处理目标变量为二分类（0/1）
	yProcessed := mat.NewDense(rows, 1, nil)
	for i := 0; i < rows; i++ {
		val := yDense.At(i, 0)
		yProcessed.Set(i, 0, map[bool]float64{true: 1, false: 0}[val > 0])
	}

	// 添加偏置项
	XWithBias := mat.NewDense(rows, cols+1, nil)
	for i := 0; i < rows; i++ {
		XWithBias.Set(i, 0, 1.0)
		for j := 0; j < cols; j++ {
			XWithBias.Set(i, j+1, xDense.At(i, j))
		}
	}

	// 训练逻辑回归
	weights := trainLogisticRegression(XWithBias, yProcessed, 0.01, 1000)

	// 计算重要性
	importances := make([]FeatureImportance, cols)
	maxWeight := 0.0
	for j := 0; j < cols; j++ {
		if w := math.Abs(weights[j+1]); w > maxWeight {
			maxWeight = w
		}
	}
	for j, name := range featureNames {
		imp := math.Abs(weights[j+1])
		if maxWeight > 0 {
			imp /= maxWeight
		}
		importances[j] = FeatureImportance{
			Feature:    name,
			Importance: imp,
			Type:       "logistic_regression",
		}
	}

	return importances
}

// 线性回归特征重要性（回归问题）
func linearRegressionImportance(X, y mat.Matrix, featureNames []string) []FeatureImportance {
	log.Printf("开始线性回归特征重要性计算")
	xDense := ensureDenseMatrix(X)
	yDense := ensureDenseMatrix(y)
	if xDense == nil || yDense == nil {
		log.Printf("线性回归：矩阵转换失败")
		return nil
	}

	rows, cols := xDense.Dims()

	// 添加偏置项
	XWithBias := mat.NewDense(rows, cols+1, nil)
	for i := 0; i < rows; i++ {
		XWithBias.Set(i, 0, 1.0)
		for j := 0; j < cols; j++ {
			XWithBias.Set(i, j+1, xDense.At(i, j))
		}
	}

	// 训练线性回归
	weights := trainLinearRegression(XWithBias, yDense)

	// 标准化权重
	scaledWeights := scaleWeightsByFeatureStd(xDense, weights[1:])

	// 计算重要性
	importances := make([]FeatureImportance, cols)
	maxWeight := 0.0
	for j := 0; j < cols; j++ {
		if w := math.Abs(scaledWeights[j]); w > maxWeight {
			maxWeight = w
		}
	}
	for j, name := range featureNames {
		imp := math.Abs(scaledWeights[j])
		if maxWeight > 0 {
			imp /= maxWeight
		}
		importances[j] = FeatureImportance{
			Feature:    name,
			Importance: imp,
			Type:       "linear_regression",
		}
	}

	return importances
}

// 互信息特征重要性
// func mutualInformationImportance(X, y mat.Matrix, featureNames []string, taskType TaskType) []FeatureImportance {
// 	xDense := ensureDenseMatrix(X)
// 	yDense := ensureDenseMatrix(y)
// 	if xDense == nil || yDense == nil {
// 		log.Printf("互信息：矩阵转换失败")
// 		return nil
// 	}
//
// 	rows, cols := xDense.Dims()
// 	importances := make([]FeatureImportance, cols)
//
// 	// 简化实现：模拟互信息计算
// 	for i, name := range featureNames {
// 		importances[i] = FeatureImportance{
// 			Feature:    name,
// 			Importance: rand.Float64(), // 实际应计算真实互信息
// 			Type:       "mutual_information",
// 		}
// 	}
// 	return importances
// }

// 训练逻辑回归（梯度下降）
func trainLogisticRegression(X, y *mat.Dense, learningRate float64, iterations int) []float64 {
	rows, cols := X.Dims()
	weights := make([]float64, cols)

	for iter := 0; iter < iterations; iter++ {
		predictions := mat.NewDense(rows, 1, nil)
		predictions.Mul(X, mat.NewDense(cols, 1, weights))
		sigmoid(predictions)

		error := mat.NewDense(rows, 1, nil)
		error.Sub(predictions, y)

		gradient := mat.NewDense(1, cols, nil)
		gradient.Mul(error.T(), X)
		gradient.Scale(learningRate/float64(rows), gradient)

		gradVals := gradient.RawRowView(0)
		for j := 0; j < cols; j++ {
			weights[j] -= gradVals[j]
		}
	}

	return weights
}

// 训练线性回归（最小二乘法）
func trainLinearRegression(X, y *mat.Dense) []float64 {
	_, cols := X.Dims()
	XT := mat.Dense{}
	XT.CloneFrom(X.T())

	XTX := mat.Dense{}
	XTX.Mul(&XT, X)

	XTXInv := mat.Dense{}
	if err := XTXInv.Inverse(&XTX); err != nil {
		log.Printf("矩阵求逆失败: %v", err)
		return make([]float64, cols)
	}

	XTY := mat.Dense{}
	XTY.Mul(&XT, y)

	weights := mat.Dense{}
	weights.Mul(&XTXInv, &XTY)

	weightsSlice := make([]float64, cols)
	for i := 0; i < cols; i++ {
		weightsSlice[i] = weights.At(i, 0)
	}

	return weightsSlice
}

// 辅助函数：确保矩阵是*mat.Dense类型
func ensureDenseMatrix(m mat.Matrix) *mat.Dense {
	if m == nil {
		return nil
	}
	// 尝试直接类型转换
	if dense, ok := m.(*mat.Dense); ok {
		return dense
	}
	// 转换失败则复制数据到新的Dense矩阵
	rows, cols := m.Dims()
	data := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			data[i*cols+j] = m.At(i, j)
		}
	}
	return mat.NewDense(rows, cols, data)
}

// 辅助函数：sigmoid激活函数
func sigmoid(m *mat.Dense) {
	rows, cols := m.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := m.At(i, j)
			m.Set(i, j, 1.0/(1.0+math.Exp(-val)))
		}
	}
}

// 辅助函数：权重标准化（线性回归）
func scaleWeightsByFeatureStd(X *mat.Dense, weights []float64) []float64 {
	rows, cols := X.Dims()
	scaled := make([]float64, cols)

	for j := 0; j < cols; j++ {
		feature := make([]float64, rows)
		for i := 0; i < rows; i++ {
			feature[i] = X.At(i, j)
		}
		std := stat.StdDev(feature, nil)
		if std > 0 {
			scaled[j] = weights[j] * std
		}
	}

	return scaled
}

// 这里需要你根据实际数据格式实现convertToMatrix函数
// 功能：将字符串记录转换为数值矩阵X（特征）和y（目标）
// func convertToMatrix(headers []string, records [][]string, featureIndices []int, targetIdx int) (X, y mat.Matrix, taskType TaskType) {
// 	// 示例实现（需要根据你的数据格式修改）
// 	rows := len(records)
// 	if rows == 0 {
// 		return nil, nil, ""
// 	}
// 	cols := len(featureIndices)
//
// 	// 解析特征矩阵X
// 	xData := make([]float64, rows*cols)
// 	for i, record := range records {
// 		for j, idx := range featureIndices {
// 			val, err := parseFloat(record[idx])
// 			if err != nil {
// 				log.Printf("解析特征失败: 行=%d, 列=%d, 值=%s", i, idx, record[idx])
// 				val = 0
// 			}
// 			xData[i*cols+j] = val
// 		}
// 	}
//
// 	// 解析目标变量y
// 	yData := make([]float64, rows)
// 	isClassification := false
// 	classSet := make(map[string]bool)
// 	for i, record := range records {
// 		val, err := parseFloat(record[targetIdx])
// 		if err != nil {
// 			// 尝试作为分类变量处理
// 			classSet[record[targetIdx]] = true
// 			val = float64(len(classSet)) // 简单映射为数值
// 			isClassification = true
// 		}
// 		yData[i] = val
// 	}
//
// 	// 确定任务类型（如果类别数较少则视为分类）
// 	if !isClassification {
// 		taskType = TaskRegression
// 	} else if len(classSet) <= 10 { // 阈值可调整
// 		taskType = TaskClassification
// 	} else {
// 		taskType = TaskRegression
// 	}
//
// 	return mat.NewDense(rows, cols, xData), mat.NewDense(rows, 1, yData), taskType
// }

// 辅助函数：字符串转float64
func parseFloat(s string) (float64, error) {
	// 实现字符串到float64的转换（处理可能的空值等）
	// 这里使用标准库函数，你可以根据需要扩展
	return strconv.ParseFloat(s, 64)
}

// 注意：需要导入以下包

// 初始化随机数种子
func init() {
	rand.Seed(time.Now().UnixNano())
}

// package main
//
// import (
// 	"gonum.org/v1/gonum/mat"
// 	"gonum.org/v1/gonum/stat"
// 	"math"
// 	"log"
// )
//
// // FeatureImportance 特征重要性结果结构
// // type FeatureImportance struct {
// // 	Feature    string  `json:"feature"`
// // 	Importance float64 `json:"importance"`
// // 	Type       string  `json:"type"`
// // }
//
// // 逻辑回归特征重要性（分类问题）
// func logisticRegressionImportance(X, y mat.Matrix, featureNames []string) []FeatureImportance {
// 	// 将mat.Matrix转换为*mat.Dense
// 	log.Printf("begin logisticRegressionImportance" )
// 	xDense, okX := X.(*mat.Dense)
// 	yDense, okY := y.(*mat.Dense)
// 	if !okX || !okY {
// 		return nil // 类型转换失败
// 	}
//
// 	rows, cols := xDense.Dims()
//
// 	// 1. 处理目标变量：确保是二分类（0/1）
// 	yProcessed := mat.NewDense(rows, 1, nil)
// 	for i := 0; i < rows; i++ {
// 		val := yDense.At(i, 0)
// 		if val > 0 {
// 			yProcessed.Set(i, 0, 1)
// 		} else {
// 			yProcessed.Set(i, 0, 0)
// 		}
// 	}
//
// 	// 2. 添加偏置项（截距）
// 	XWithBias := mat.NewDense(rows, cols+1, nil)
// 	for i := 0; i < rows; i++ {
// 		XWithBias.Set(i, 0, 1.0) // 偏置项
// 		for j := 0; j < cols; j++ {
// 			XWithBias.Set(i, j+1, xDense.At(i, j))
// 		}
// 	}
//
// 	// 3. 训练逻辑回归模型（使用梯度下降）
// 	weights := trainLogisticRegression(XWithBias, yProcessed, 0.01, 1000)
//
// 	// 4. 提取特征重要性（排除偏置项）
// 	importances := make([]FeatureImportance, cols)
// 	maxWeight := 0.0
// 	for j := 0; j < cols; j++ {
// 		absWeight := math.Abs(weights[j+1])
// 		if absWeight > maxWeight {
// 			maxWeight = absWeight
// 		}
// 	}
//
// 	for j := 0; j < cols; j++ {
// 		absWeight := math.Abs(weights[j+1])
// 		importance := absWeight
// 		if maxWeight > 0 {
// 			importance = absWeight / maxWeight
// 		}
// 		importances[j] = FeatureImportance{
// 			Feature:    featureNames[j],
// 			Importance: importance,
// 			Type:       "logistic_regression",
// 		}
// 	}
//
// 	return importances
// }
//
// // 线性回归特征重要性（回归问题）
// func linearRegressionImportance(X, y mat.Matrix, featureNames []string) []FeatureImportance {
// 	// 将mat.Matrix转换为*mat.Dense
// 	log.Printf("begin linearRegressionImportance" )
// 	xDense, okX := X.(*mat.Dense)
// 	yDense, okY := y.(*mat.Dense)
// 	if !okX || !okY {
// 	    log.Printf("类型转换失败" )
// 		return nil // 类型转换失败
// 	}
//
// 	rows, cols := xDense.Dims()
//
// 	// 2. 添加偏置项（截距）
// 	XWithBias := mat.NewDense(rows, cols+1, nil)
// 	for i := 0; i < rows; i++ {
// 		XWithBias.Set(i, 0, 1.0) // 偏置项
// 		for j := 0; j < cols; j++ {
// 			XWithBias.Set(i, j+1, xDense.At(i, j))
// 		}
// 	}
//
// 	// 3. 训练线性回归模型（使用最小二乘法）
// 	weights := trainLinearRegression(XWithBias, yDense)
//
// 	// 4. 标准化特征（用于正确比较权重）
// 	scaledWeights := scaleWeightsByFeatureStd(xDense, weights[1:])
//
// 	// 5. 提取特征重要性
// 	importances := make([]FeatureImportance, cols)
// 	maxWeight := 0.0
// 	for j := 0; j < cols; j++ {
// 		absWeight := math.Abs(scaledWeights[j])
// 		if absWeight > maxWeight {
// 			maxWeight = absWeight
// 		}
// 	}
//
// 	for j := 0; j < cols; j++ {
// 		absWeight := math.Abs(scaledWeights[j])
// 		importance := absWeight
// 		if maxWeight > 0 {
// 			importance = absWeight / maxWeight
// 		}
// 		importances[j] = FeatureImportance{
// 			Feature:    featureNames[j],
// 			Importance: importance,
// 			Type:       "linear_regression",
// 		}
// 	}
//     log.Printf("importances:%s",importances )
// 	return importances
// }
//
// // 训练逻辑回归模型（梯度下降）
// func trainLogisticRegression(X, y *mat.Dense, learningRate float64, iterations int) []float64 {
// 	rows, cols := X.Dims()
// 	weights := make([]float64, cols)
//
// 	for iter := 0; iter < iterations; iter++ {
// 		predictions := mat.NewDense(rows, 1, nil)
// 		predictions.Mul(X, mat.NewDense(cols, 1, weights))
// 		sigmoid(predictions)
//
// 		error := mat.NewDense(rows, 1, nil)
// 		error.Sub(predictions, y)
//
// 		gradient := mat.NewDense(1, cols, nil)
// 		gradient.Mul(error.T(), X)
// 		gradient.Scale(learningRate/float64(rows), gradient)
//
// 		gradVals := gradient.RawRowView(0)
// 		for j := 0; j < cols; j++ {
// 			weights[j] -= gradVals[j]
// 		}
// 	}
//
// 	return weights
// }
//
// // 训练线性回归模型（最小二乘法）
// func trainLinearRegression(X, y *mat.Dense) []float64 {
// 	// 获取矩阵维度（修正Cols()方法错误）
// 	_, cols := X.Dims() // 使用Dims()获取列数，而不是Cols()
//
// 	// 计算 (X^T X)^-1 X^T y（修正矩阵转置方法）
// 	XT := mat.Dense{}
// 	XT.CloneFrom(X.T()) // 正确的矩阵转置方法
//
// 	XTX := mat.Dense{}
// 	XTX.Mul(&XT, X)
//
// 	XTXInv := mat.Dense{}
// 	err := XTXInv.Inverse(&XTX)
// 	if err != nil {
// 		return make([]float64, cols)
// 	}
//
// 	XTY := mat.Dense{}
// 	XTY.Mul(&XT, y)
//
// 	weights := mat.Dense{}
// 	weights.Mul(&XTXInv, &XTY)
//
// 	// 提取权重值
// 	weightsSlice := make([]float64, cols)
// 	for i := 0; i < cols; i++ {
// 		weightsSlice[i] = weights.At(i, 0)
// 	}
//
// 	return weightsSlice
// }
//
// // 辅助函数：应用sigmoid函数
// func sigmoid(m *mat.Dense) {
// 	rows, cols := m.Dims()
// 	for i := 0; i < rows; i++ {
// 		for j := 0; j < cols; j++ {
// 			val := m.At(i, j)
// 			m.Set(i, j, 1.0/(1.0+math.Exp(-val)))
// 		}
// 	}
// }
//
// // 辅助函数：通过特征标准差缩放权重
// func scaleWeightsByFeatureStd(X *mat.Dense, weights []float64) []float64 {
// 	rows, cols := X.Dims()
// 	scaled := make([]float64, cols)
//
// 	for j := 0; j < cols; j++ {
// 		feature := make([]float64, rows)
// 		for i := 0; i < rows; i++ {
// 			feature[i] = X.At(i, j)
// 		}
// 		std := stat.StdDev(feature, nil)
//
// 		if std > 0 {
// 			scaled[j] = weights[j] * std
// 		} else {
// 			scaled[j] = 0
// 		}
// 	}
//
// 	return scaled
// }