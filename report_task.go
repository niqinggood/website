package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"log"
	"os"
	"path/filepath"
	"reflect"
	"time"
)

// 全局临时结构体定义
type tempSectionData struct {
	Title   string      // 小节标题
	Content string      // 注释内容
	SQL     string      // SQL语句（仅内部使用，不显示在HTML中）
	Data    interface{} // SQL执行结果
}

// 生成报表任务主函数
// 生成报表任务主函数
func PDataExecuteReportTask(task *PDataTask) (interface{}, error) {
	// 1. 解析报表配置（从Specific["report"]提取）
	var reportConfig struct {
		BodySections []struct {
			Title   string `json:"title"`
			Content string `json:"content"`
			SQL     string `json:"sql"`
		} `json:"bodySections"`
	}

	var specificMap map[string]json.RawMessage
	if err := json.Unmarshal(task.Specific, &specificMap); err != nil {
		return nil, fmt.Errorf("解析Specific失败: %v", err)
	}
	reportRaw, ok := specificMap["report"]
	if !ok {
		return nil, errors.New("Specific中未找到report配置")
	}
	if err := json.Unmarshal(reportRaw, &reportConfig); err != nil {
		return nil, fmt.Errorf("解析report配置失败: %v", err)
	}

	// 验证配置
	if len(reportConfig.BodySections) == 0 {
		return nil, errors.New("报表内容段落不能为空")
	}
	for i, section := range reportConfig.BodySections {
		if section.Content == "" && section.SQL == "" {
			return nil, fmt.Errorf("段落 %d 必须填写内容或SQL语句", i+1)
		}
	}

	// 2. 执行SQL并收集数据（使用全局db）
	var sections []tempSectionData
	for i, sec := range reportConfig.BodySections {
		log.Printf("处理段落 %d: %s", i+1, sec.Title)
		tempSec := tempSectionData{
			Title:   sec.Title,
			Content: sec.Content,
			SQL:     sec.SQL,
		}

		// 执行真实SQL查询
		if sec.SQL != "" {
			var result []map[string]interface{}
			if err := db.Raw(sec.SQL).Scan(&result).Error; err != nil {
				return nil, fmt.Errorf("段落 %d SQL执行失败: %v", i+1, err)
			}
			tempSec.Data = result
		}
		sections = append(sections, tempSec)
	}

	// 3. 生成HTML报表（保存到./data/report目录）
	filePath, err := generateReportHTML(task, sections)
	if err != nil {
		return nil, fmt.Errorf("生成HTML报表失败: %v", err)
	}

	// 4. 生成带域名/IP和端口的下载链接
	downloadURL, err := getFullDownloadURL(filePath)
	if err != nil {
		return nil, fmt.Errorf("生成下载链接失败: %v", err)
	}

	return map[string]interface{}{
		"single_file":    downloadURL, // 完整下载链接（含域名/IP和端口）
		"sections_count": len(reportConfig.BodySections),
		"generated_at":   time.Now().Format(time.RFC3339),
		"message":        "报表生成成功",
	}, nil
}

func generateReportHTML(task *PDataTask, sections []tempSectionData) (string, error) {
	reportDir := "./data/report"
	if err := os.MkdirAll(reportDir, 0755); err != nil {
		return "", fmt.Errorf("创建报表目录失败: %w", err)
	}

	fileName := fmt.Sprintf("report_%s_%s.html",
		task.ID,
		time.Now().Format("20060102150405"),
	)
	filePath := filepath.Join(reportDir, fileName)

	// 优化后的HTML模板 - 更专业美观
	const htmlTemplate = `
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{.TaskName}} - 数据分析报表</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            /* 专业柔和的配色方案 */
            --primary: #6685e2ff;         /* 主色：更柔和的蓝色 */
            --primary-light: #f0f4ff;   /* 主色浅背景 */
            --primary-ultralight: #f7f9ff; 
            --text: #2d3748;           /* 文本色：深灰 */
            --text-light: #718096;      /* 次要文本色 */
            --text-lighter: #a0aec0;    /* 更浅的文本色 */
            --border: #e2e8f0;          /* 边框色 */
            --bg-light: #f8fafc;        /* 浅背景色 */
            --bg-section: #ffffff;      /* 区块背景色 */
            --shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            --shadow-hover: 0 4px 6px rgba(0, 0, 0, 0.04);
            --radius: 8px;              /* 圆角 */
            --transition: all 0.2s ease;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', 'Noto Sans SC', sans-serif;
            line-height: 1.6;
            color: var(--text);
            background-color: var(--bg-light);
            padding: 24px 16px;
        }

        .container {
            max-width: 1100px;
            margin: 0 auto;
        }

        /* 顶部区块优化 - 更简洁专业 */
        .report-header {
            background-color: var(--bg-section);
            padding: 24px;
            border-radius: var(--radius);
            margin-bottom: 28px;
            box-shadow: var(--shadow);
            border: 1px solid var(--border);
        }

        .report-title {
            font-size: 1.4rem;
            font-weight: 600;
            color: var(--primary);
            margin-bottom: 8px;
        }

        .report-subtitle {
            font-size: 0.95rem;
            color: var(--text-light);
            margin-bottom: 20px;
        }

        /* 元信息网格 */
        .meta-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
        }

        .meta-item {
            padding: 12px;
            background-color: var(--bg-light);
            border-radius: 6px;
        }

        .meta-label {
            font-size: 0.8rem;
            color: var(--text-lighter);
            margin-bottom: 4px;
        }

        .meta-value {
            font-size: 0.95rem;
            font-weight: 500;
        }

        /* 内容区块容器 */
        .sections-container {
            display: grid;
            gap: 20px;
        }

        /* 单个区块样式 */
        .section {
            background: var(--bg-section);
            border-radius: var(--radius);
            padding: 24px;
            box-shadow: var(--shadow);
            border: 1px solid var(--border);
            transition: var(--transition);
        }

        .section:hover {
            box-shadow: var(--shadow-hover);
        }

        /* 区块头部 */
        .section-header {
            display: flex;
            align-items: center;
            margin-bottom: 18px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
        }

        .section-number {
            width: 28px;
            height: 28px;
            background-color: var(--primary);
            color: white;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 500;
            font-size: 0.85rem;
            margin-right: 12px;
            flex-shrink: 0;
        }

        .section-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text);
            margin: 0;
        }

        /* 区块内容 */
        .section-content {
            color: var(--text-light);
            margin-bottom: 18px;
            line-height: 1.7;
            font-size: 0.95rem;
        }

        .section-content p {
            margin-bottom: 10px;
        }

        /* 数据容器 */
        .data-wrapper {
            overflow-x: auto;
            margin-top: 16px;
            border-radius: var(--radius);
        }

        /* 表格样式 */
        .data-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
            box-shadow: var(--shadow);
            border-radius: var(--radius);
            overflow: hidden;
        }

        .data-table th {
            background-color: var(--primary-light);
            color: var(--primary);
            padding: 12px 16px;
            text-align: left;
            font-weight: 500;
            border-bottom: 1px solid var(--border);
        }

        .data-table td {
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
            font-size: 0.88rem;
        }

        .data-table tr:last-child td {
            border-bottom: none;
        }

        .data-table tr:hover td {
            background-color: var(--primary-light);
        }

        /* 非表格数据样式 */
        .data-card {
            background-color: var(--primary-ultralight);
            border-left: 3px solid var(--primary);
            padding: 16px;
            border-radius: var(--radius);
        }

        .data-card-title {
            font-size: 0.9rem;
            color: var(--primary);
            margin-bottom: 8px;
            font-weight: 500;
        }

        .data-card-content {
            background-color: white;
            padding: 12px;
            border-radius: 4px;
            border: 1px solid var(--border);
            font-family: 'SFMono-Regular', Consolas, monospace;
            font-size: 0.85rem;
            white-space: pre-wrap;
            word-break: break-all;
        }

        /* 响应式适配 */
        @media (max-width: 768px) {
            .report-header {
                padding: 20px;
            }
            
            .report-title {
                font-size: 1.2rem;
            }
            
            .meta-grid {
                grid-template-columns: 1fr;
            }
            
            .section {
                padding: 20px;
            }
            
            .section-title {
                font-size: 1rem;
            }
        }

        /* 动画优化 */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .section {
            animation: fadeIn 0.4s ease-out forwards;
            opacity: 0;
        }

        .section:nth-child(1) { animation-delay: 0.05s; }
        .section:nth-child(2) { animation-delay: 0.1s; }
        .section:nth-child(3) { animation-delay: 0.15s; }
        .section:nth-child(n+4) { animation-delay: 0.2s; }
    </style>
</head>
<body>
    <div class="container">
        <header class="report-header">
            <h1 class="report-title">{{.TaskName}}</h1>
            <p class="report-subtitle">数据分析报表 · 生成于 {{.GenerateTime}}</p>
            
            <div class="meta-grid">
                <div class="meta-item">
                    <div class="meta-label">任务ID</div>
                    <div class="meta-value">{{.TaskID}}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">区块数量</div>
                    <div class="meta-value">{{len .Sections}} 个</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">报表状态</div>
                    <div class="meta-value">生成成功</div>
                </div>
            </div>
        </header>

        <main class="sections-container">
            {{range $index, $section := .Sections}}
            <section class="section">
                <div class="section-header">
                    <div class="section-number">{{add $index 1}}</div>
                    <h2 class="section-title">{{.Title}}</h2>
                </div>
                
                {{if .Content}}
                <div class="section-content">
                    {{.Content}}
                </div>
                {{end}}

                {{if .Data}}
                <div class="data-wrapper">
                    {{if isArray .Data}}
                        {{$first := index .Data 0}}
                        {{if isMap $first}}
                        <table class="data-table">
                            <thead>
                                <tr>
                                    {{range $key, $_ := $first}}
                                    <th>{{$key}}</th>
                                    {{end}}
                                </tr>
                            </thead>
                            <tbody>
                                {{range .Data}}
                                <tr>
                                    {{range $value := .}}
                                    <td>{{$value}}</td>
                                    {{end}}
                                </tr>
                                {{end}}
                            </tbody>
                        </table>
                        {{else}}
                        <table class="data-table">
                            <thead>
                                <tr><th>序号</th><th>值</th></tr>
                            </thead>
                            <tbody>
                                {{range $i, $item := .Data}}
                                <tr>
                                    <td>{{add $i 1}}</td>
                                    <td>{{$item}}</td>
                                </tr>
                                {{end}}
                            </tbody>
                        </table>
                        {{end}}
                    {{else if isMap .Data}}
                    <table class="data-table">
                        <thead>
                            <tr><th>键</th><th>值</th></tr>
                        </thead>
                        <tbody>
                            {{range $key, $value := .Data}}
                            <tr>
                                <th>{{$key}}</th>
                                <td>{{$value}}</td>
                            </tr>
                            {{end}}
                        </tbody>
                    </table>
                    {{else}}
                    <div class="data-card">
                        <div class="data-card-title">数据详情</div>
                        <div class="data-card-content">{{.Data}}</div>
                    </div>
                    {{end}}
                </div>
                {{end}}
            </section>
            {{end}}
        </main>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // 打印优化
            window.addEventListener('keydown', (e) => {
                if (e.ctrlKey && e.key === 'p') {
                    document.querySelectorAll('.section').forEach(sec => {
                        sec.style.animation = 'none';
                        sec.style.opacity = '1';
                    });
                }
            });
        });
    </script>
</body>
</html>
	`

	tpl, err := template.New("report").Funcs(template.FuncMap{
		"isArray": func(v interface{}) bool {
			if v == nil {
				return false
			}
			kind := reflect.TypeOf(v).Kind()
			return kind == reflect.Slice || kind == reflect.Array
		},
		"isMap": func(v interface{}) bool {
			if v == nil {
				return false
			}
			return reflect.TypeOf(v).Kind() == reflect.Map
		},
		"add": func(a, b int) int {
			return a + b
		},
	}).Parse(htmlTemplate)
	if err != nil {
		return "", fmt.Errorf("模板解析失败: %w", err)
	}

	templateData := struct {
		TaskName     string
		TaskID       string
		GenerateTime string
		Sections     []tempSectionData
	}{
		TaskName:     task.TaskName,
		TaskID:       task.ID,
		GenerateTime: time.Now().Format("2006-01-02 15:04:05"),
		Sections:     sections,
	}

	var htmlBuf bytes.Buffer
	if err := tpl.Execute(&htmlBuf, templateData); err != nil {
		return "", fmt.Errorf("模板渲染失败: %w", err)
	}

	if err := os.WriteFile(filePath, htmlBuf.Bytes(), 0644); err != nil {
		return "", fmt.Errorf("写入报表文件失败: %w", err)
	}

	log.Printf("报表生成成功，路径: %s", filePath)
	return filePath, nil
}

// // 生成HTML报表并保存到./data/report目录
// func generateReportHTML(task *PDataTask, sections []tempSectionData) (string, error) {
// 	// 定义报表存储目录：当前目录下的data/report
// 	reportDir := "./data/report"
// 	if err := os.MkdirAll(reportDir, 0755); err != nil {
// 		return "", fmt.Errorf("创建报表目录失败: %v", err)
// 	}

// 	// 生成唯一文件名：任务ID+时间戳（确保不重复）
// 	fileName := fmt.Sprintf("report_%s_%s.html", task.ID, time.Now().Format("20060102150405"))
// 	filePath := filepath.Join(reportDir, fileName)

// 	// HTML模板（仅显示标题、内容和结果）
// 	const htmlTemplate = `
// <!DOCTYPE html>
// <html lang="zh-CN">
// <head>
//     <meta charset="UTF-8">
//     <title>{{.TaskName}}</title>
//     <style>
//         body { font-family: "Microsoft YaHei", sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; color: #333; }
//         .report-header { border-bottom: 3px solid #2c3e50; padding: 15px 0; margin-bottom: 30px; }
//         .report-title { color: #2c3e50; margin: 0; font-size: 24px; }
//         .meta-info { color: #666; margin-top: 10px; font-size: 14px; }
//         .section { margin-bottom: 40px; padding: 20px; background: #f9f9f9; border-radius: 8px; }
//         .section-title { color: #3498db; margin-top: 0; border-left: 4px solid #3498db; padding-left: 10px; }
//         .section-content { margin: 15px 0; line-height: 1.6; }
//         .data-table { width: 100%; border-collapse: collapse; margin-top: 15px; }
//         .data-table th { background: #e9f5ff; padding: 10px; text-align: left; border: 1px solid #ddd; }
//         .data-table td { padding: 10px; border: 1px solid #ddd; }
//         .data-table tr:hover { background: #f5f9ff; }
//     </style>
// </head>
// <body>
//     <div class="report-header">
//         <h1 class="report-title">{{.TaskName}}</h1>
//         <div class="meta-info">
//             任务ID: {{.TaskID}} | 生成时间: {{.GenerateTime}}
//         </div>
//     </div>

//     {{range .Sections}}
//     <div class="section">
//         <h2 class="section-title">{{.Title}}</h2>

//         {{if .Content}}
//         <div class="section-content">{{.Content}}</div>
//         {{end}}

//         {{if .Data}}
//         <div>

//             {{if isArray .Data}}
//             <table class="data-table">
//                 {{$first := index .Data 0}}
//                 {{if isMap $first}}
//                 <tr>{{range $key, $_ := $first}}<th>{{$key}}</th>{{end}}</tr>
//                 {{range .Data}}<tr>{{range $value := .}}<td>{{$value}}</td>{{end}}</tr>{{end}}
//                 {{else}}
//                 <tr><th>数据</th></tr>
//                 {{range .Data}}<tr><td>{{.}}</td></tr>{{end}}
//                 {{end}}
//             </table>
//             {{else if isMap .Data}}
//             <table class="data-table">
//                 {{range $key, $value := .Data}}<tr><th>{{$key}}</th><td>{{$value}}</td></tr>{{end}}
//             </table>
//             {{else}}
//             <div class="section-content">{{.Data}}</div>
//             {{end}}
//         </div>
//         {{end}}
//     </div>
//     {{end}}
// </body>
// </html>
// 	`

// 	// 渲染模板
// 	tpl, err := template.New("report").Funcs(template.FuncMap{
// 		"isArray": func(v interface{}) bool {
// 			return reflect.TypeOf(v).Kind() == reflect.Slice || reflect.TypeOf(v).Kind() == reflect.Array
// 		},
// 		"isMap": func(v interface{}) bool {
// 			return reflect.TypeOf(v).Kind() == reflect.Map
// 		},
// 	}).Parse(htmlTemplate)
// 	if err != nil {
// 		return "", fmt.Errorf("模板解析失败: %v", err)
// 	}

// 	// 模板数据
// 	templateData := struct {
// 		TaskName     string
// 		TaskID       string
// 		GenerateTime string
// 		Sections     []tempSectionData
// 	}{
// 		TaskName:     task.TaskName,
// 		TaskID:       task.ID,
// 		GenerateTime: time.Now().Format("2006-01-02 15:04:05"),
// 		Sections:     sections,
// 	}

// 	// 渲染并写入文件
// 	var htmlBuf bytes.Buffer
// 	if err := tpl.Execute(&htmlBuf, templateData); err != nil {
// 		return "", fmt.Errorf("模板渲染失败: %v", err)
// 	}
// 	if err := os.WriteFile(filePath, htmlBuf.Bytes(), 0644); err != nil {
// 		return "", fmt.Errorf("写入报表文件失败: %v", err)
// 	}

// 	log.Printf("报表已保存至: %s", filePath)
// 	return filePath, nil
// }

// 获取带域名/IP和端口的完整下载链接
func getFullDownloadURL(localFilePath string) (string, error) {
	// 1. 获取服务IP（优先使用flag配置的api_ip，否则自动获取本机IP）
	// serverIP, err := *api_ip
	// if err != nil {
	// 	return "", err
	// }

	// 2. 提取文件在data/report下的相对路径（用于下载路由）

	// 生成文件名（格式：配置名_任务ID_时间戳.csv）
	// if configName == "" {
	// 	configName = "result"
	// }  , configName string

	log.Printf("Ip:%s  port:%s", server_ip, server_port)
	downloadURL := fmt.Sprintf("http://%s:%s/download/%s", server_ip, server_port, localFilePath)
	return downloadURL, nil
}

// 辅助函数：取最小值
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
