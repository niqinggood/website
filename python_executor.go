// package main
package main

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

type PythonProcess struct {
	cmd    *exec.Cmd
	stdin  *bufio.Writer
	stdout *bufio.Reader
	mu     sync.Mutex
}

// 转义特殊字符：处理引号、反斜杠等可能破坏参数解析的字符
func escapeCode(code string) string {
	// 替换双引号为转义双引号
	code = strings.ReplaceAll(code, `"`, `\"`)
	// 替换反斜杠为双反斜杠
	code = strings.ReplaceAll(code, `\`, `\\`)
	// 替换换行符为转义换行（可选，根据需求决定是否保留换行）
	// code = strings.ReplaceAll(code, "\n", `\n`)
	return code
}

// 启动 Python 进程
func startPythonProcess(load_file string, code string, start_node string) (*PythonProcess, error) {
	log.Println("begin startPythonProcess")
	cmd1 := exec.Command("python", "--version")
	out, err := cmd1.Output()
	if err != nil {
		fmt.Println("Python 环境问题:", err)
	} else {
		fmt.Println("Python 环境:", string(out))
	}

	pythonCmd := "python3"
	if _, err := exec.LookPath("python"); err != nil {
		// 如果 python 不存在，尝试 python3
		if _, err := exec.LookPath("python3"); err != nil {
			return nil, fmt.Errorf("无法找到 python 或 python3 可执行文件")
		}
		pythonCmd = "python"
	}
	escapedCode := escapeCode(code)
	cmd := exec.Command(pythonCmd, "-m", "saddle.comm.task_dispatch",
		"--task", "inference",
		"--stdin",
		"--load_file", load_file,
		"--infer_proccode", escapedCode,
		"--start_node", start_node)
	cmdStr := cmd.String()
	log.Println("cmd:", cmdStr)

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("无法获取 stdin: %v", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("无法获取 stdout: %v", err)
	}
	stderr, err := cmd.StderrPipe() // 捕获错误输出
	if err != nil {
		return nil, fmt.Errorf("无法获取 stderr: %v", err)
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("无法启动 Python 进程: %v", err)
	}
	go func() {
		reader := bufio.NewReaderSize(stderr, 3*1024*1024) // 1MB缓冲区
		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				if err != io.EOF {
					log.Printf("读取 stderr 时出错: %v", err)
				}
				break
			}
			log.Printf("Python stderr: %s", strings.TrimSpace(line))
		}
	}()

	return &PythonProcess{
		cmd:    cmd,
		stdin:  bufio.NewWriter(stdin),
		stdout: bufio.NewReaderSize(stdout, 30*1024*1024), //bufio.NewReader(stdout),
	}, nil
}

func sendFrame(w io.Writer, msgType uint32, data []byte) error { // 改为uint32
	header := make([]byte, 8)
	binary.BigEndian.PutUint32(header[0:4], msgType)           // 类型 (4字节)
	binary.BigEndian.PutUint32(header[4:8], uint32(len(data))) // 长度 (4字节)

	// 计算CRC（仅对header+data计算）
	crc := crc32.ChecksumIEEE(append(header, data...))

	buf := bytes.NewBuffer(header)
	buf.Write(data)
	binary.Write(buf, binary.BigEndian, crc) // 追加CRC (4字节)

	_, err := w.Write(buf.Bytes())
	return err
}

// 读取帧
func readFrame(r io.Reader) (byte, []byte, error) {
	// 读取类型+长度（8字节）
	header := make([]byte, 8)
	if _, err := io.ReadFull(r, header); err != nil {
		return 0, nil, err
	}

	msgType := header[0]
	length := binary.BigEndian.Uint32(header[4:8])

	// 读取数据+CRC
	frame := make([]byte, length+4)
	if _, err := io.ReadFull(r, frame); err != nil {
		return 0, nil, err
	}

	// 校验CRC
	crc := binary.BigEndian.Uint32(frame[length:])
	if crc32.ChecksumIEEE(append(header, frame[:length]...)) != crc {
		return 0, nil, errors.New("CRC校验失败")
	}

	return msgType, frame[:length], nil
}

// func (p *PythonProcess) runInference(dataSource map[string]interface{}) ([]byte, error) {
// 	log.Printf("begin runInference ")
// 	// p.mu.Lock()
// 	log.Printf("p mu locked ")
// 	// defer p.mu.Unlock()

// 	// 1. 构造请求
// 	log.Printf("Constructing request with dataSource: %+v", dataSource)
// 	request := map[string]interface{}{"dataSource": dataSource}

// 	requestJSON, err := json.Marshal(request)
// 	if err != nil {
// 		log.Printf("JSON marshaling failed: %v, input: %+v", err, request)
// 		return nil, fmt.Errorf("请求构造失败: %v", err)
// 	}
// 	log.Printf("Request JSON constructed: %s", string(requestJSON))

// 	// 2. 发送增强协议消息
// 	log.Printf("Sending frame with type 0x01 (request)")
// 	if err := sendFrame(p.stdin, 0x01, requestJSON); err != nil {
// 		log.Printf("Frame sending failed: %v", err)
// 		return nil, fmt.Errorf("发送请求失败: %v", err)
// 	}
// 	log.Printf("Frame sent successfully")

// 	// 3. 读取响应
// 	log.Printf("Waiting for response frame...")
// 	msgType, data, err := readFrame(p.stdout)
// 	if err != nil {
// 		log.Printf("Frame reading failed: %v", err)
// 		return nil, fmt.Errorf("读取响应失败: %v", err)
// 	}
// 	log.Printf("Received frame type: 0x%x, data length: %d", msgType, len(data))

// 	// 4. 验证消息类型
// 	if msgType != 0x02 {
// 		errMsg := fmt.Sprintf("非法响应类型: 0x%x (expected 0x02)", msgType)
// 		log.Printf(errMsg)
// 		return nil, fmt.Errorf(errMsg)
// 	}

// 	log.Printf("Inference completed successfully")
// 	return data, nil
// }

// func (p *PythonProcess) runInference(dataSource map[string]interface{}) (*string, error) {
// 	p.mu.Lock()

// 	// 构造请求
// 	request := map[string]interface{}{
// 		"dataSource": dataSource,
// 	}
// 	requestJSON, err := json.Marshal(request)
// 	if err != nil {
// 		log.Printf("序列化请求失败: %v", err)
// 		return nil, fmt.Errorf("序列化请求失败: %v", err)
// 	}
// 	log.Println("发送请求:", truncateString(string(requestJSON), 100))

// 	// 发送请求
// 	if _, err := p.stdin.WriteString(string(requestJSON) + "\n"); err != nil {
// 		log.Printf("发送请求失败: %v", err)
// 		return nil, fmt.Errorf("发送请求失败: %v", err)
// 	}
// 	log.Println("请求发送成功")
// 	p.stdin.Flush()

// 	// 读取结果
// 	log.Println("等待读取结果...")
// 	// reader := bufio.NewReader(p.stdout)
// 	// 读取 stdout 结果（使用大缓冲区）
// 	// responseJSON, err := reader.ReadString('\n')
// 	reader := bufio.NewReaderSize(p.stdout, 30*1024*1024) // 1MB 缓冲区
// 	var responseJSON strings.Builder

// 	// 持续读取直到遇到自定义结束标记
// 	for {
// 		log.Println("for_read")
// 		line, err := reader.ReadString('\n')
// 		if err != nil {
// 			return nil, fmt.Errorf("读取失败: %v", err)
// 		}

// 		// 检查是否到达结束标记
// 		if strings.HasSuffix(line, "EndbyPython\n") {
// 			responseJSON.WriteString(strings.TrimSuffix(line, "EndbyPython\n"))
// 			break
// 		}

// 		responseJSON.WriteString(line)
// 	}
// 	result := responseJSON.String()
// 	log.Println("收到结果:", truncateString(result, 100))

// 	p.mu.Unlock()
// 	return &result, nil
// }

func (p *PythonProcess) runInference(dataSource map[string]interface{}) (*string, error) {
	p.mu.Lock()
	defer func() {
		p.mu.Unlock()
		log.Println("已释放互斥锁")
	}()

	// 构造请求

	log.Println("开始构造请求数据...")
	var requestJSON []byte
	var err error

	// 正确判断preInferString是否在dataSource中（原代码错误判断了request，应该是dataSource）
	if preInferStr, ok := dataSource["preInferString"]; !ok {
		// 不存在时，构建request并序列化
		request := map[string]interface{}{
			"dataSource": dataSource,
		}
		requestJSON, err = json.Marshal(request)
		if err != nil {
			log.Printf("序列化请求失败: %v", err)
			return nil, fmt.Errorf("序列化请求失败: %v", err)
		}
	} else {
		// 存在时，进行类型转换（确保是字符串类型）
		preInferStrVal, ok := preInferStr.(string)
		if !ok {
			log.Printf("preInferString不是字符串类型")
			return nil, fmt.Errorf("preInferString不是字符串类型")
		}
		// 转换为[]byte赋值给requestJSON（json.Marshal的结果类型）
		requestJSON = []byte(preInferStrVal)
	}
	log.Printf("请求数据构造完成，长度: %d 字节", len(requestJSON))
	log.Println("请求内容(截断):", truncateString(string(requestJSON), 100))

	// 发送请求
	log.Println("开始发送请求...")
	if _, err := p.stdin.WriteString(string(requestJSON) + "\n"); err != nil {
		log.Printf("发送请求失败: %v", err)
		return nil, fmt.Errorf("发送请求失败: %v", err)
	}
	if err := p.stdin.Flush(); err != nil {
		log.Printf("刷新缓冲区失败: %v", err)
		return nil, fmt.Errorf("刷新缓冲区失败: %v", err)
	}
	log.Println("请求发送成功")

	// 读取结果
	log.Println("开始等待响应...")
	reader := bufio.NewReaderSize(p.stdout, 30*1024*1024)
	var responseJSON strings.Builder
	readTimeout := time.After(70 * time.Second) // 30秒超时
	lineCount := 0
OUTER_LOOP:
	for {
		select {
		case <-readTimeout:
			log.Println("错误：读取响应超时")
			return nil, fmt.Errorf("读取响应超时")
		default:
			line, err := reader.ReadString('\n')
			log.Println("ReadString 调用返回，错误状态:", err)
			lineCount++
			log.Printf("读取到第 %d 行数据，长度: %d 字节", lineCount, len(line))

			if err != nil {
				if err == io.EOF {
					log.Println("警告：到达数据流末尾但未找到结束标记")
				} else {
					log.Printf("读取错误: %v", err)
				}
				return nil, fmt.Errorf("读取失败: %v", err)
			}

			// 调试输出原始行内容
			//log.Printf("原始行数据(前100字符): %q", truncateString(line, 100))
			log.Printf("收到原始行: %q", line) // 原始数据验证
			// 修改结束标记检测逻辑（兼容 \n 和 \r\n）
			if strings.HasSuffix(line, "EndbyPython\n") || strings.HasSuffix(line, "EndbyPython\r\n") {
				// 统一处理换行符
				log.Printf("find end") // 原始数据验证
				line = strings.TrimSuffix(line, "EndbyPython\n")
				line = strings.TrimSuffix(line, "EndbyPython\r\n")
				responseJSON.WriteString(line)
				log.Printf("当前累积数据大小: %d 字节", responseJSON.Len())
				break OUTER_LOOP
			}
			//responseJSON.WriteString(line)
			log.Printf("当前累积数据大小: %d 字节", responseJSON.Len())
		}
	}
	log.Printf(" continue ")

	result := responseJSON.String()
	log.Printf("收到完整响应，总长度: %d 字节", len(result))
	log.Println("响应内容(截断):", truncateString(result, 100))

	return &result, nil
}

// 辅助函数：截断字符串用于显示
func truncateString(s string, maxLen int) string {
	if len(s) > maxLen {
		return s[:maxLen] + "...(总长度:" + strconv.Itoa(len(s)) + ")"
	}
	return s
}

// 执行Python处理进程（保持之前的实现）
func executeImageChainPythonProcess(recordID uint, imagePath string, stepsJSON string, annotations interface{}) (error, string) {
	logger.Infof("开始启动Python进程 [Record%d]", recordID)

	// 确定Python命令
	pythonCmd := "python3"
	possiblePythonCommands := []string{"python3", "python"}
	if runtime.GOOS == "windows" {
		possiblePythonCommands = []string{"py", "python", "python3", "python.exe", "python3.exe"}
	}

	found := false
	for _, cmd := range possiblePythonCommands {
		if _, err := exec.LookPath(cmd); err == nil {
			pythonCmd = cmd
			found = true
			break
		}
	}

	if !found {
		return fmt.Errorf("无法找到Python可执行文件"), ""
	}

	workDir := fmt.Sprintf("projects/chain_process/%d", recordID)
	if err := os.MkdirAll(workDir, 0755); err != nil {
		return fmt.Errorf("创建工作目录失败: %v", err), ""
	}

	log.Println("workDir:", workDir)

	// 创建工作目录

	// 定义结果文件和日志文件路径
	resultFilePath := filepath.Join(workDir, "results.json")
	logPath := filepath.Join(workDir, "process.log")

	// 准备参数
	annotationsJSON, _ := json.Marshal(annotations)

	// escapedSteps := escapeCode(stepsJSON)
	stepsFilePath := filepath.Join(workDir, "steps.json")

	// 将steps信息保存到文件
	if err := os.WriteFile(stepsFilePath, []byte(stepsJSON), 0644); err != nil {
		return fmt.Errorf("保存步骤文件失败: %v", err), ""
	}
	logger.Infof("步骤文件已保存: %s", stepsFilePath)

	// 构建命令
	cmd := exec.Command(pythonCmd, "-m", "saddle.comm.imagechain",
		"--task", "offline",
		"--input_path", imagePath,
		"--steps_filepath", stepsFilePath, // 改为文件路径
		"--annotations", string(annotationsJSON),
		"--resultfile_path", resultFilePath,
		"--log_path", logPath,
		"--record_id", fmt.Sprintf("%d", recordID))

	cmdStr := cmd.String()

	// 执行命令并获取输出
	output, err := cmd.CombinedOutput()
	outputStr := string(output)

	logger.Infof("Python命令 [Record%d]: %s", recordID, cmdStr)
	logger.Infof("Python脚本输出 [Record%d]:\n%s", recordID, outputStr)

	// 读取并记录日志文件内容
	if logContent, err := os.ReadFile(logPath); err == nil {
		logger.Infof("Python处理日志 [Record%d]:\n%s", recordID, string(logContent))
	} else {
		logger.Warnf("无法读取日志文件 [Record%d]: %v", recordID, err)
	}

	// 解析结果文件
	resultData, err := parseResultFile(resultFilePath)
	if err != nil {
		return fmt.Errorf("解析结果文件失败: %v", err), outputStr
	}

	// 汇总结果
	finalResult := map[string]interface{}{
		"process_output": outputStr,
		"result_data":    resultData,
		"log_path":       logPath,
		"result_path":    resultFilePath,
		"record_id":      recordID,
	}

	// 解析日志文件内容
	logContent, logErr := os.ReadFile(logPath)
	if logErr != nil {
		if os.IsNotExist(logErr) {
			finalResult["log_content"] = "no log file"
		} else {
			finalResult["log_content"] = fmt.Sprintf("read log error: %v", logErr)
		}
	} else {
		finalResult["log_content"] = string(logContent)
	}

	// 转换为JSON字符串
	resultJSON, err := json.Marshal(finalResult)
	if err != nil {
		return fmt.Errorf("序列化结果失败: %v", err), outputStr
	}

	return err, string(resultJSON)
}

func parseResultFile(resultFilePath string) (interface{}, error) {
	// 检查结果文件是否存在
	if _, err := os.Stat(resultFilePath); os.IsNotExist(err) {
		return nil, fmt.Errorf("结果文件不存在: %s", resultFilePath)
	}

	// 读取结果文件
	resultContent, err := os.ReadFile(resultFilePath)
	if err != nil {
		return nil, fmt.Errorf("读取结果文件失败: %v", err)
	}

	// 如果文件为空，返回空结果
	if len(resultContent) == 0 {
		return map[string]interface{}{"status": "empty_result"}, nil
	}

	// 尝试解析为JSON
	var resultData interface{}
	if err := json.Unmarshal(resultContent, &resultData); err != nil {
		// 如果无法解析为JSON，返回原始内容
		return string(resultContent), nil
	}

	return resultData, nil
}

func executeTablePipelinePythonProcess(recordID string, mainTable TableData, auxiliaryTables []TableData, processingSteps []StepConfig, outputConfig OutputConfig) (string, error) {
	logger.Infof("开始启动Python表格处理进程 [Record%s]", recordID)

	// 确定Python命令
	pythonCmd := "python3"
	possiblePythonCommands := []string{"python3", "python"}
	if runtime.GOOS == "windows" {
		possiblePythonCommands = []string{"py", "python", "python3", "python.exe", "python3.exe"}
	}

	found := false
	for _, cmd := range possiblePythonCommands {
		if _, err := exec.LookPath(cmd); err == nil {
			pythonCmd = cmd
			found = true
			break
		}
	}

	if !found {
		return "", fmt.Errorf("无法找到Python可执行文件")
	}

	// 创建工作目录
	workDir := fmt.Sprintf("./projects/table_pipeline/%s", recordID)
	if err := os.MkdirAll(workDir, 0755); err != nil {
		return "", fmt.Errorf("创建工作目录失败: %v", err)
	}

	logger.Infof("工作目录: %s", workDir)

	// 定义文件路径
	resultFilePath := filepath.Join(workDir, "results.json")
	logPath := filepath.Join(workDir, "process.log")

	// 准备表格元数据和样例数据
	tableMetadata := map[string]interface{}{
		"main_table": map[string]interface{}{
			"name":        mainTable.Name,
			"file_path":   mainTable.FileName,
			"columns":     mainTable.Columns,
			"sample_data": getSampleData(mainTable.PreviewData, 3),
			"total_rows":  len(mainTable.FullData),
			"file_type":   getFileType(mainTable.FileName),
			"filehash":    mainTable.Filehash,
		},
		"auxiliary_tables": make(map[string]interface{}),
	}

	// 添加辅助表信息
	for _, auxTable := range auxiliaryTables {
		tableMetadata["auxiliary_tables"].(map[string]interface{})[auxTable.Name] = map[string]interface{}{
			"name":        auxTable.Name,
			"file_path":   auxTable.FileName,
			"columns":     auxTable.Columns,
			"sample_data": getSampleData(auxTable.PreviewData, 3),
			"total_rows":  len(auxTable.FullData),
			"file_type":   getFileType(auxTable.FileName),
			"filehash":    mainTable.Filehash,
		}
	}

	// 保存处理配置到文件
	config := map[string]interface{}{
		"table_metadata":   tableMetadata,
		"processing_steps": processingSteps,
		"output_config":    outputConfig,
		"record_id":        recordID,
		"work_dir":         workDir,
	}

	configJSON, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return "", fmt.Errorf("序列化配置失败: %v", err)
	}

	configFilePath := filepath.Join(workDir, "pipeline_config.json")
	if err := os.WriteFile(configFilePath, configJSON, 0644); err != nil {
		return "", fmt.Errorf("保存配置文件失败: %v", err)
	}
	logger.Infof("配置文件已保存: %s", configFilePath)

	// 构建命令
	cmd := exec.Command(pythonCmd, "-m", "saddle.comm.tablepipeline",
		"--config_file", configFilePath,
		"--result_file", resultFilePath,
		"--log_path", logPath)

	// 设置工作目录为项目根目录
	cmd.Dir = "."

	// 执行命令并获取输出
	output, err := cmd.CombinedOutput()
	outputStr := string(output)

	logger.Infof("Python表格处理命令 [Record%s]: %s", recordID, cmd.String())
	logger.Infof("Python表格处理脚本输出 [Record%s]:\n%s", recordID, outputStr)

	// 读取并记录日志文件内容
	var logContent string
	if content, err := os.ReadFile(logPath); err == nil {
		logContent = string(content)
		logger.Infof("Python表格处理日志 [Record%s]:\n%s", recordID, logContent)
	} else {
		logContent = fmt.Sprintf("无法读取日志文件: %v", err)
		logger.Warnf("无法读取日志文件 [Record%s]: %v", recordID, err)
	}

	// 解析结果文件
	resultData, resultErr := parseTableResultFile(resultFilePath)
	if resultErr != nil {
		logger.Errorf("解析结果文件失败 [Record%s]: %v", recordID, resultErr)
	}

	// 汇总结果返回给前端
	finalResult := map[string]interface{}{
		"process_output": outputStr,
		"result_data":    resultData,
		"log_content":    logContent,
		"log_path":       logPath,
		"result_path":    resultFilePath,
		"record_id":      recordID,
	}

	// 如果Python处理失败，返回错误
	if err != nil {
		finalResult["error"] = err.Error()
		return toJSONString(finalResult), err
	}

	if resultErr != nil {
		finalResult["error"] = resultErr.Error()
		return toJSONString(finalResult), resultErr
	}

	// 处理成功，构建前端需要的响应格式
	if resultData != nil {
		if resultMap, ok := resultData.(map[string]interface{}); ok {
			// 构建步骤执行结果
			stepResults := buildStepResults(processingSteps, resultMap)
			finalResult["step_results"] = stepResults

			// 构建最终输出
			finalResult["final_output"] = resultMap["output_file"]
		}
	}

	return toJSONString(finalResult), nil
}

func getSampleData(fullData []map[string]interface{}, maxRows int) []map[string]interface{} {
	if len(fullData) == 0 {
		return []map[string]interface{}{}
	}

	if len(fullData) <= maxRows {
		return fullData
	}

	return fullData[:maxRows]
}

func getFileType(filename string) string {
	ext := strings.ToLower(filepath.Ext(filename))
	switch ext {
	case ".csv":
		return "csv"
	case ".xlsx", ".xls":
		return "excel"
	case ".json":
		return "json"
	default:
		return "unknown"
	}
}

func buildFinalOutput(resultMap map[string]interface{}, workDir string, format string) map[string]interface{} {
	finalOutput := map[string]interface{}{
		"output_file": filepath.Join(workDir, fmt.Sprintf("output.%s", format)),
	}

	if outputData, exists := resultMap["output_data"]; exists {
		if outputMap, ok := outputData.(map[string]interface{}); ok {
			if columns, exists := outputMap["columns"]; exists {
				finalOutput["columns"] = columns
			}
			if dataLength, exists := outputMap["data_length"]; exists {
				finalOutput["data_length"] = dataLength
			} else if rows, exists := outputMap["rows"]; exists {
				if rowSlice, ok := rows.([]interface{}); ok {
					finalOutput["data_length"] = len(rowSlice)
				}
			}
			if sampleData, exists := outputMap["sample_data"]; exists {
				finalOutput["sample_data"] = sampleData
			} else if rows, exists := outputMap["rows"]; exists {
				if rowSlice, ok := rows.([]interface{}); ok {
					sampleSize := 3
					if len(rowSlice) < sampleSize {
						sampleSize = len(rowSlice)
					}
					finalOutput["sample_data"] = rowSlice[:sampleSize]
				}
			}
		}
	}

	return finalOutput
}

func parseTableResultFile(resultFilePath string) (interface{}, error) {
	if _, err := os.Stat(resultFilePath); os.IsNotExist(err) {
		return nil, fmt.Errorf("结果文件不存在: %s", resultFilePath)
	}

	resultContent, err := os.ReadFile(resultFilePath)
	if err != nil {
		return nil, fmt.Errorf("读取结果文件失败: %v", err)
	}

	if len(resultContent) == 0 {
		return map[string]interface{}{"status": "empty_result"}, nil
	}

	var resultData interface{}
	if err := json.Unmarshal(resultContent, &resultData); err != nil {
		return string(resultContent), nil
	}

	return resultData, nil
}

func buildStepResults(steps []StepConfig, resultMap map[string]interface{}) []StepResult {
	var stepResults []StepResult

	for i, step := range steps {
		stepResult := StepResult{
			StepID:        step.ID,
			Success:       true,
			Message:       fmt.Sprintf("Step %d (%s) executed successfully", i+1, step.Type),
			OutputColumns: []string{},
			SampleData:    []map[string]interface{}{},
			DataLength:    0,
			ExecutionTime: "0.5s",
		}

		stepKey := fmt.Sprintf("step_%d", i)
		if stepData, exists := resultMap[stepKey]; exists {
			if stepMap, ok := stepData.(map[string]interface{}); ok {
				if success, ok := stepMap["success"].(bool); ok {
					stepResult.Success = success
				}
				if message, ok := stepMap["message"].(string); ok {
					stepResult.Message = message
				}
				if columns, ok := stepMap["output_columns"].([]interface{}); ok {
					for _, col := range columns {
						if colStr, ok := col.(string); ok {
							stepResult.OutputColumns = append(stepResult.OutputColumns, colStr)
						}
					}
				}
				if sample, ok := stepMap["sample_data"].([]interface{}); ok {
					for _, item := range sample {
						if data, ok := item.(map[string]interface{}); ok {
							stepResult.SampleData = append(stepResult.SampleData, data)
						}
					}
				}
				if length, ok := stepMap["data_length"].(float64); ok {
					stepResult.DataLength = int(length)
				}
				if execTime, ok := stepMap["execution_time"].(string); ok {
					stepResult.ExecutionTime = execTime
				}
			}
		}

		stepResults = append(stepResults, stepResult)
	}

	return stepResults
}

// // 发送请求并获取结果
// func (p *PythonProcess) runInference(dataSource map[string]interface{}) (map[string]interface{}, error) {
// 	p.mu.Lock()
// 	defer p.mu.Unlock()

// 	// 构造请求
// 	request := map[string]interface{}{
// 		"dataSource": dataSource,
// 	}
// 	requestJSON, err := json.Marshal(request)
// 	if err != nil {
// 		return nil, fmt.Errorf("序列化请求失败: %v", err)
// 	}

// 	// 发送请求
// 	if _, err := p.stdin.WriteString(string(requestJSON) + "\n"); err != nil {
// 		return nil, fmt.Errorf("发送请求失败: %v", err)
// 	}
// 	p.stdin.Flush()

// 	// 读取结果
// 	responseJSON, err := p.stdout.ReadString('\n')
// 	if err != nil {
// 		return nil, fmt.Errorf("读取结果失败: %v", err)
// 	}

// 	// 解析结果
// 	var result map[string]interface{}
// 	if err := json.Unmarshal([]byte(responseJSON), &result); err != nil {
// 		return nil, fmt.Errorf("解析结果失败: %v", err)
// 	}

// 	return result, nil
// }

// import (
// 	"bufio"
// 	"encoding/json"
// 	"fmt"
// 	"os/exec"
// 	"sync"
// )

// type PythonProcess struct {
// 	cmd    *exec.Cmd
// 	stdin  *bufio.Writer
// 	stdout *bufio.Reader
// 	mu     sync.Mutex
// }

// // 启动 Python 进程
// func startPythonProcess() (*PythonProcess, error) {
// 	cmd := exec.Command("python", "task_dispatch.py")

// 	stdin, err := cmd.StdinPipe()
// 	if err != nil {
// 		return nil, fmt.Errorf("无法获取 stdin: %v", err)
// 	}
// 	stdout, err := cmd.StdoutPipe()
// 	if err != nil {
// 		return nil, fmt.Errorf("无法获取 stdout: %v", err)
// 	}

// 	if err := cmd.Start(); err != nil {
// 		return nil, fmt.Errorf("无法启动 Python 进程: %v", err)
// 	}

// 	return &PythonProcess{
// 		cmd:    cmd,
// 		stdin:  bufio.NewWriter(stdin),
// 		stdout: bufio.NewReader(stdout),
// 	}, nil
// }

// // 发送请求并获取结果
// func (p *PythonProcess) runInference(dataSource map[string]interface{}) (map[string]interface{}, error) {
// 	p.mu.Lock()
// 	defer p.mu.Unlock()

// 	// 构造请求
// 	request := map[string]interface{}{
// 		"dataSource": dataSource,
// 	}
// 	requestJSON, err := json.Marshal(request)
// 	if err != nil {
// 		return nil, fmt.Errorf("序列化请求失败: %v", err)
// 	}

// 	// 发送请求
// 	if _, err := p.stdin.WriteString(string(requestJSON) + "\n"); err != nil {
// 		return nil, fmt.Errorf("发送请求失败: %v", err)
// 	}
// 	p.stdin.Flush()

// 	// 读取结果
// 	responseJSON, err := p.stdout.ReadString('\n')
// 	if err != nil {
// 		return nil, fmt.Errorf("读取结果失败: %v", err)
// 	}

// 	// 解析结果
// 	var result map[string]interface{}
// 	if err := json.Unmarshal([]byte(responseJSON), &result); err != nil {
// 		return nil, fmt.Errorf("解析结果失败: %v", err)
// 	}

// 	return result, nil
// }
