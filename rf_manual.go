package main

import (

	"math/rand"
	"gonum.org/v1/gonum/mat"
	"time"
)

// FeatureImportance 特征重要性结果结构
// type FeatureImportance struct {
// 	Feature    string  `json:"feature"`
// 	Importance float64 `json:"importance"`
// 	Type       string  `json:"type"`
// }

// 随机森林分类特征重要性（纯Go实现）
func randomForestClassificationImportance(X, y mat.Matrix, featureNames []string) []FeatureImportance {
	rand.Seed(time.Now().UnixNano())
	rows, cols := X.Dims()
	numTrees := 30
	sampleSize := int(float64(rows) * 0.8)

	importances := make([]float64, cols)

	// 训练多棵决策树
	for t := 0; t < numTrees; t++ {
		// 随机采样数据
		sampledIndices := sampleWithReplacement(rows, sampleSize)
		treeImp := buildClassificationTree(X, y, sampledIndices, cols)

		// 累加重要性
		for i := 0; i < cols; i++ {
			importances[i] += treeImp[i]
		}
	}

	// 平均并归一化
	result := make([]FeatureImportance, cols)
	maxImp := maxSlice(importances)
	for i, name := range featureNames {
		normImp := importances[i] / float64(numTrees)
		if maxImp > 0 {
			normImp /= maxImp // 归一化到[0,1]
		}
		result[i] = FeatureImportance{
			Feature:    name,
			Importance: normImp,
			Type:       "rf_classification",
		}
	}

	return result
}

// 随机森林回归回归特征重要性（纯Go实现）
func randomForestRegressionImportance(X, y mat.Matrix, featureNames []string) []FeatureImportance {
	rand.Seed(time.Now().UnixNano())
	rows, cols := X.Dims()
	numTrees := 30
	sampleSize := int(float64(rows) * 0.8)

	importances := make([]float64, cols)

	for t := 0; t < numTrees; t++ {
		sampledIndices := sampleWithReplacement(rows, sampleSize)
		treeImp := buildRegressionTree(X, y, sampledIndices, cols)

		for i := 0; i < cols; i++ {
			importances[i] += treeImp[i]
		}
	}

	result := make([]FeatureImportance, cols)
	maxImp := maxSlice(importances)
	for i, name := range featureNames {
		normImp := importances[i] / float64(numTrees)
		if maxImp > 0 {
			normImp /= maxImp
		}
		result[i] = FeatureImportance{
			Feature:    name,
			Importance: normImp,
			Type:       "rf_regression",
		}
	}

	return result
}

// 辅助函数：有放回采样
func sampleWithReplacement(total, size int) []int {
	samples := make([]int, size)
	for i := 0; i < size; i++ {
		samples[i] = rand.Intn(total)
	}
	return samples
}

// 构建分类树并计算特征重要性
func buildClassificationTree(X, y mat.Matrix, samples []int, numFeatures int) []float64 {
	importances := make([]float64, numFeatures)
	// 随机选择部分特征
	selectedFeatures := randomFeatures(numFeatures, 0.7)

	// 计算每个特征的Gini增益（简化实现）
	for _, f := range selectedFeatures {
		gain := calculateGiniGain(X, y, samples, f)
		importances[f] = gain
	}

	return importances
}

// 构建回归树并计算特征重要性
func buildRegressionTree(X, y mat.Matrix, samples []int, numFeatures int) []float64 {
	importances := make([]float64, numFeatures)
	selectedFeatures := randomFeatures(numFeatures, 0.7)

	// 计算每个特征的方差减少
	for _, f := range selectedFeatures {
		reduction := calculateVarianceReduction(X, y, samples, f)
		importances[f] = reduction
	}

	return importances
}

// 随机选择特征
func randomFeatures(total int, ratio float64) []int {
	count := int(float64(total) * ratio)
	if count < 1 {
		count = 1
	}

	features := make(map[int]bool)
	for len(features) < count {
		f := rand.Intn(total)
		features[f] = true
	}

	result := make([]int, 0, count)
	for f := range features {
		result = append(result, f)
	}
	return result
}

// 计算Gini增益（分类）
func calculateGiniGain(X, y mat.Matrix, samples []int, feature int) float64 {
	// 获取特征值和对应的标签
	values := make([]float64, len(samples))
	labels := make([]float64, len(samples))
	for i, idx := range samples {
		values[i] = X.At(idx, feature)
		labels[i] = y.At(idx, 0)
	}

	// 简化实现：使用随机分割点计算增益
	splitVal := findMedian(values)
	leftLabels, rightLabels := splitByThreshold(labels, values, splitVal)

	// 计算父节点不纯度
	parentGini := giniImpurity(labels)

	// 计算子节点不纯度
	leftGini := giniImpurity(leftLabels)
	rightGini := giniImpurity(rightLabels)

	// 计算增益
	pLeft := float64(len(leftLabels)) / float64(len(labels))
	pRight := 1 - pLeft
	gain := parentGini - (pLeft*leftGini + pRight*rightGini)

	return gain
}

// 计算方差减少（回归）
func calculateVarianceReduction(X, y mat.Matrix, samples []int, feature int) float64 {
	values := make([]float64, len(samples))
	targets := make([]float64, len(samples))
	for i, idx := range samples {
		values[i] = X.At(idx, feature)
		targets[i] = y.At(idx, 0)
	}

	splitVal := findMedian(values)
	leftTargets, rightTargets := splitByThreshold(targets, values, splitVal)

	parentVariance := variance(targets)
	leftVariance := variance(leftTargets)
	rightVariance := variance(rightTargets)

	pLeft := float64(len(leftTargets)) / float64(len(targets))
	pRight := 1 - pLeft

	reduction := parentVariance - (pLeft*leftVariance + pRight*rightVariance)
	return reduction
}

// 辅助函数：计算中位数
func findMedian(values []float64) float64 {
	// 简化实现
	if len(values) == 0 {
		return 0
	}
	return values[len(values)/2]
}

// 辅助函数：根据阈值分割数据
func splitByThreshold(targets, values []float64, threshold float64) (left, right []float64) {
	for i, v := range values {
		if v <= threshold {
			left = append(left, targets[i])
		} else {
			right = append(right, targets[i])
		}
	}
	return
}

// 辅助函数：计算Gini不纯度
func giniImpurity(labels []float64) float64 {
	if len(labels) == 0 {
		return 0
	}

	// 计算类别频率
	freq := make(map[float64]int)
	for _, l := range labels {
		freq[l]++
	}

	// 计算Gini
	gini := 1.0
	for _, count := range freq {
		p := float64(count) / float64(len(labels))
		gini -= p * p
	}
	return gini
}

// 辅助函数：计算方差
func variance(values []float64) float64 {
	if len(values) < 2 {
		return 0
	}

	mean := 0.0
	for _, v := range values {
		mean += v
	}
	mean /= float64(len(values))

	variance := 0.0
	for _, v := range values {
		variance += (v - mean) * (v - mean)
	}
	return variance / float64(len(values))
}

// 辅助函数：求最大值
func maxSlice(slice []float64) float64 {
	maxVal := 0.0
	for _, v := range slice {
		if v > maxVal {
			maxVal = v
		}
	}
	return maxVal
}
