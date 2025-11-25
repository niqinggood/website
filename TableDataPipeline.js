import React, { useState, useEffect } from 'react';
import {
  Table, Input, Select, Button, InputNumber, Space, message, Form,
  Card, List, Upload, Divider, Row, Col, Alert, Typography, Collapse,
  Tag, Modal, Switch, Progress, Tooltip, Badge, Radio
} from 'antd';
import {
  PlusOutlined, DeleteOutlined, ArrowUpOutlined, ArrowDownOutlined,
  UploadOutlined, EyeOutlined, TableOutlined, EditOutlined,
  SettingOutlined, ColumnHeightOutlined, ExportOutlined,DownloadOutlined,
  CodeOutlined, DatabaseOutlined, RocketOutlined, PlayCircleOutlined,ReloadOutlined,LinkOutlined,FilterOutlined,ForkOutlined
} from '@ant-design/icons';
import {  FileTextOutlined } from '@ant-design/icons';
import axios from 'axios';
import { StarFilled, StarOutlined } from '@ant-design/icons';
import ReactJson from '@microlink/react-json-view';
import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import { TypeTransformConfig, RenameConfig, SampleConfig, DeduplicateConfig,
         SortConfig,AppendConfig,ConfigWrapper }  from './tableOperator';
import MergeDataConfig      from './MergeDataConfig'
import DataEditOperator     from './DataEdit'
import FilterDataConfig     from './FilterDataConfig'
import TableTypeConfig      from './TableTypeConfig'
const { Title, Text } = Typography;
const { Panel } = Collapse;
const { Option } = Select;
// 科技感主题颜色
const techColors = {
  primary: '#1890ff',
  success: '#52c41a',
  warning: '#faad14',
  error: '#ff4d4f',
  info: '#13c2c2',
  purple: '#722ed1',
  cyan: '#13c2c2'
};

const getFileUrl = (filePath) => {
  if (!filePath) return '';

  // 如果已经是完整URL，直接返回
  if (filePath.startsWith('http://') || filePath.startsWith('https://')) {
    return filePath;
  }

  // 获取baseURL
  let baseURL = '';
  try {
    baseURL = axios.defaults.baseURL || window.config?.apiPilotBaseUrl || '';
  } catch (error) {
    baseURL = '';
  }

  // 移除baseURL末尾的斜杠（如果有）
  if (baseURL.endsWith('/')) {
    baseURL = baseURL.slice(0, -1);
  }

  // 确保filePath不以斜杠开头
  let cleanFilePath = filePath;
  if (cleanFilePath.startsWith('/')) {
    cleanFilePath = cleanFilePath.slice(1);
  }

  return `${baseURL}/download/${cleanFilePath}`;
};

const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

const parseUploadedFile = (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = (e) => {
      try {
        const data = e.target.result;
        const tables = [];

        if (file.name.includes('.xlsx') || file.name.includes('.xls')) {
          // 处理 Excel 文件
          const workbook = XLSX.read(data, { type: 'binary' });

          workbook.SheetNames.forEach((sheetName, index) => {
            const worksheet = workbook.Sheets[sheetName];
            const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });

            if (jsonData.length > 0) {
              const headers = jsonData[0];
              const rows = jsonData.slice(1).map((row, rowIndex) => {
                const obj = {};
                headers.forEach((header, colIndex) => {
                  obj[header] = row[colIndex] !== undefined ? row[colIndex] : null;
                });
                return { ...obj, key: rowIndex };
              });

              tables.push({
                id: `table_${Date.now()}_${index}`,
                name: `${file.name}$${sheetName}`,
                fileName: file.name,
                type: index === 0 ? 'main' : 'auxiliary',
                columns: headers,
                previewData: rows.slice(0, 10), // 只取前10行作为预览
                fullData: rows // 保存完整数据
              });
            }
          });
        } else if (file.name.includes('.csv')) {
          // 处理 CSV 文件
          Papa.parse(data, {
            header: true,
            complete: (results) => {
              const rows = results.data.filter(row =>
                Object.values(row).some(value => value !== undefined && value !== '')
              );

              tables.push({
                id: `table_${Date.now()}_0`,
                name: file.name, //.replace(/\.[^/.]+$/, "")
                fileName: file.name,
                type: 'main',
                columns: results.meta.fields || [],
                previewData: rows.slice(0, 10),
                fullData: rows
              });

              resolve(tables);
            },
            error: (error) => reject(error)
          });
          return;
        } else if (file.name.includes('.json')) {
          // 处理 JSON 文件
          const jsonData = JSON.parse(data);
          const isArray = Array.isArray(jsonData);
          const rows = isArray ? jsonData : [jsonData];

          if (rows.length > 0) {
            const columns = Object.keys(rows[0]);

            tables.push({
              id: `table_${Date.now()}_0`,
              name: file.name.replace(/\.[^/.]+$/, ""),
              fileName: file.name,
              type: 'main',
              columns: columns,
              previewData: rows.slice(0, 10),
              fullData: rows
            });
          }
        }

        resolve(tables);
      } catch (error) {
        reject(error);
      }
    };

    reader.onerror = () => reject(new Error('文件读取失败'));

    if (file.name.includes('.xlsx') || file.name.includes('.xls')) {
      reader.readAsBinaryString(file);
    } else {
      reader.readAsText(file);
    }
  });
};


// 数据预览组件
const DataPreview = ({ data, title, columns, fileName, onExport }) => {
  const [previewMode, setPreviewMode] = useState('table');
  const [isCollapsed, setIsCollapsed] = useState(true);

  if (!data || data.length === 0) {
    return (
      <Card
        title={
          <Space>
            <DatabaseOutlined />
            数据预览
          </Space>
        }
        style={{
          marginBottom: 16,
          background: 'linear-gradient(135deg, #fafafa 0%, #f5f5f5 100%)'
        }}
      >
        <Alert
          message="暂无数据"
          description="请上传测试文件或等待数据加载"
          type="info"
          showIcon
        />
      </Card>
    );
  }

  const previewData = isCollapsed ? data.slice(0, 2) : data.slice(0,5);
  //{fileName && <Tag color="blue" icon={<CodeOutlined />}>{fileName}</Tag>}
  return (
    <Card
      title={
        <Space>
          <DatabaseOutlined style={{ color: techColors.primary }} />
          {title || "数据预览"}
          <Badge count={data.length} style={{ backgroundColor: techColors.primary }} />
          <Badge count={`${columns.length}列`} style={{ backgroundColor: techColors.info }} />

        </Space>
      }
      extra={
        <Space>
          <Tooltip title={isCollapsed ? '展开更多数据' : '收起数据'}>
            <Button
              type="link"
              onClick={() => setIsCollapsed(!isCollapsed)}
              icon={<ColumnHeightOutlined />}
            >
              {isCollapsed ? '展开' : '收起'}
            </Button>
          </Tooltip>
        </Space>
      }
      style={{
        marginBottom: 16,
        background: 'linear-gradient(135deg, #fafafa 0%, #f5f5f5 100%)'
      }}
    >
      <>
        <Table
          dataSource={previewData.map((item, index) => ({ ...item, key: index }))}
          columns={columns.map(col => ({
            title: (
              <Tooltip title={col}>
                <Tag color="blue" style={{ margin: 0, maxWidth: 100, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {col}
                </Tag>
              </Tooltip>
            ),
            dataIndex: col,
            key: col,
            width: 120,
            render: (text) => (
              <Tooltip title={String(text)}>
                <Text style={{ fontSize: '12px' }}>
                  {typeof text === 'string' && text.length > 15 ? `${text.substring(0, 15)}...` :
                   text === null ? 'null' : text === undefined ? 'undefined' : String(text)}
                </Text>
              </Tooltip>
            )
          }))}
          pagination={false}
          size="small"
          scroll={{ x: true, y: 200 }}
          style={{ background: 'white' }}
        />
        {isCollapsed && data.length > 3 && (
          <div style={{ textAlign: 'center', marginTop: 8, padding: '8px' }}>
            <Text type="secondary">
              ... 还有 {data.length - 3} 行数据未显示，点击展开查看全部
            </Text>
          </div>
        )}
      </>
    </Card>
  );
};

// 表管理组件


const TableManager = ({ tables, onTablesChange, onRemoveTable, onSetMainTable,hidden_upload=false }) => {
  const [uploadLoading, setUploadLoading] = useState(false);
  const [activeTableTab, setActiveTableTab] = useState('main');
  const [expandedTables, setExpandedTables] = useState(new Set());
  const [lastClickTime, setLastClickTime] = useState(0); // 记录上次点击时间
  const handleTabClick = (tableId, e) => {
    const currentTime = new Date().getTime();

    // 判断是否为双击（300ms 内两次点击）
    if (currentTime - lastClickTime < 300) {
      // 双击：取消选中状态
      if (activeTableTab === tableId) {
        setActiveTableTab(''); // 清空选中状态
      }
      setLastClickTime(0); // 重置时间
      return;
    }

    // 单击：正常切换标签页
    setLastClickTime(currentTime);
    setActiveTableTab(tableId);
  };
  const UrlInputModal = ({ visible, onCancel, onOk, loading }) => {
  const [form] = Form.useForm();
  const [url, setUrl] = useState('');
  const [fileType, setFileType] = useState('auto');


  const handleOk = () => {
    form.validateFields().then(values => {
      onOk(values.url, values.fileType);
    });
  };

  const handleCancel = () => {
    form.resetFields();
    onCancel();
  };

  return (
    <Modal
      title="Load Data from URL"
      open={visible}
      onOk={handleOk}
      onCancel={handleCancel}
      confirmLoading={loading}
      okText="Load Data"
      cancelText="Cancel"
      width={600}
    >
      <Form form={form} layout="vertical">
        <Form.Item
          name="url"
          label="Data URL"
          rules={[
            { required: true, message: 'Please input the data URL' },
            { type: 'url', message: 'Please enter a valid URL' },
            {
              validator: (_, value) => {
                if (!value) return Promise.resolve();
                // 基本的 URL 验证
                try {
                  new URL(value);
                  return Promise.resolve();
                } catch {
                  return Promise.reject(new Error('Please enter a valid URL'));
                }
              }
            }
          ]}
        >
          <Input
            placeholder="https://example.com/data.csv or s3://bucket/data.csv"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            size="large"
            allowClear
          />
        </Form.Item>
        <Form.Item
          name="fileType"
          label="File Type"
          initialValue="auto"
          tooltip="Select file type or use auto-detect"
        >
          <Select size="large">
            <Option value="auto">Auto Detect</Option>
            <Option value="csv">CSV</Option>
            <Option value="excel">Excel</Option>
            <Option value="json">JSON</Option>
          </Select>
        </Form.Item>
      </Form>
      <Alert
        message="Supported Data Sources"
        description={
          <div>
            <div>• HTTP/HTTPS endpoints</div>
            <div>• Object Storage (S3, OSS, COS)</div>
            <div>• Data APIs with JSON/CSV response</div>
            <div>• Remote files up to 1GB</div>
          </div>
        }
        type="info"
        showIcon
        style={{ marginTop: 16 }}
      />
    </Modal>
  );
};

// 增强的上传组件
const EnhancedUpload = ({ onTablesChange, tables }) => {
  const [uploadLoading, setUploadLoading] = useState(false);
  const [urlModalVisible, setUrlModalVisible] = useState(false);
  const [urlLoading, setUrlLoading] = useState(false);

  // 处理文件上传（现有的逻辑）
  const handleFileUpload = async (file) => {
    setUploadLoading(true);
    try {
      const newTables = await parseUploadedFile(file);
      let tmpfilehash =null
      try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('file_name', file.name);

        const response = await axios.post('/api/tablepipeline/upload-file', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          withCredentials: true,
          timeout: 30000,
        });

        if (response.data.status === 'success') {
          message.success('File uploaded successfully');
          console.info("response.data.data",response.data.data)
          tmpfilehash = response.data.data.hash
        } else {
          throw new Error(response.data.message || 'Upload failed');
        }
      } catch (error) {
        console.error('Upload error:', error);
        let errorMessage = 'Upload failed';
        if (error.response) {
          errorMessage = error.response.data?.message || `Server error: ${error.response.status}`;
        } else if (error.request) {
          errorMessage = 'Network error, please check connection';
        } else {
          errorMessage = error.message;
        }
        message.error(errorMessage);
      }

      // 处理表名
      const processedTables = newTables.map((table, index) => {
        if (file.name.includes('.xlsx') || file.name.includes('.xls')) {
          const sheetSuffix = newTables.length > 1 ? `_sheet${index + 1}` : '';
          return {
            ...table,
            name: `${table.name}${sheetSuffix}`,
            originalSheetName: table.name,
            sourceType: 'file',
            source: file.name,
            filehash:tmpfilehash
          };
        }
        return {
          ...table,
          sourceType: 'file',
          source: file.name,
          filehash:tmpfilehash
        };
      });

      const updatedTables = [...tables, ...processedTables];
      onTablesChange(updatedTables);

      message.success(`Successfully loaded ${processedTables.length} data tables`);
    } catch (error) {
      console.error('File parsing error:', error);
      message.error(`File parsing failed: ${error.message}`);
    } finally {
      setUploadLoading(false);
    }
    return false;
  };

  // 处理 URL 数据加载
  const handleUrlLoad = async (url, fileType) => {
    setUrlLoading(true);
    try {
      const response = await axios.post('/api/tablepipeline/load-from-url', {
        url: url,
        file_type: fileType,
        preview_rows: 50, // 只获取前50行作为预览
        need_metadata: true
      }, {
        withCredentials: true,
        timeout: 60000 // 大数据可能需要更长时间
      });

      if (response.data.status === 'success') {
        const data = response.data.data;

        // 处理后端返回的表数据
        const newTables = data.tables.map((table, index) => ({
          id: `table_${Date.now()}_${index}`,
          name: table.name || `Table_${index + 1}`,
          fileName: table.file_name || url.split('/').pop() || 'remote_data',
          type: index === 0 ? 'main' : 'auxiliary',
          columns: table.columns || [],
          previewData: table.preview_data || [],
          fullData: table.full_data || table.preview_data || [],
          sourceType: 'url',
          source: url,
          originalSheetName: table.sheet_name,
          totalRows: table.total_rows,
          fileSize: table.file_size,
          loadedRows: table.loaded_rows || table.preview_data?.length || 0,
          filehash: table.file_hash,
          filepath: table.file_path
        }));

        const updatedTables = [...tables, ...newTables];
        onTablesChange(updatedTables);

        setUrlModalVisible(false);
        message.success(
          `Successfully loaded ${newTables.length} table${newTables.length > 1 ? 's' : ''} from URL` +
          (data.total_rows ? ` (${data.total_rows.toLocaleString()} rows)` : '')
        );
      } else {
        throw new Error(response.data.message || 'Failed to load data from URL');
      }
    } catch (error) {
      console.error('URL loading error:', error);
      let errorMessage = 'Failed to load data from URL';
      if (error.response) {
        errorMessage = error.response.data?.message || `Server error: ${error.response.status}`;
      } else if (error.request) {
        errorMessage = 'Network error, please check the URL and connection';
      } else {
        errorMessage = error.message;
      }
      message.error(errorMessage);
    } finally {
      setUrlLoading(false);
    }
  };



  return (
    <>
      <Space>
        <Upload
          accept=".csv,.xlsx,.xls,.json"
          showUploadList={false}
          beforeUpload={handleFileUpload}
        >
          <Button
            icon={<UploadOutlined />}
            loading={uploadLoading}
            type="primary"
          >
            Upload File
          </Button>
        </Upload>

        <Button
          icon={<LinkOutlined />}
          loading={urlLoading}
          onClick={() => setUrlModalVisible(true)}
          type="default"
          style={{
            background: 'linear-gradient(135deg, #52c41a, #73d13d)',
            border: 'none',
            color: 'white'
          }}
        >
          Load from URL
        </Button>

        {mainTable && (
          <Alert
            message="Main table for processing, auxiliary tables for merge operations"
            type="info"
            showIcon
            style={{ display: 'inline-flex', alignItems: 'center' }}
          />
        )}
      </Space>

      <UrlInputModal
        visible={urlModalVisible}
        onCancel={() => setUrlModalVisible(false)}
        onOk={handleUrlLoad}
        loading={urlLoading}
      />
    </>
  );
};


  const toggleTableExpansion = (tableId) => {
    const newExpanded = new Set(expandedTables);
    if (newExpanded.has(tableId)) {
      newExpanded.delete(tableId);
    } else {
      newExpanded.add(tableId);
    }
    setExpandedTables(newExpanded);
  };

  // 获取当前主表
  const mainTable = tables.find(t => t.type === 'main');
  const allTables = tables.filter(Boolean);

  // 判断表是否是当前主表的辅助函数
  const isMainTable = (table) => {
    return mainTable && table.id === mainTable.id;
  };

  return (
    <Card
      title={
        <Space>
          <DatabaseOutlined />
          Table Management
          <Badge count={allTables.length} showZero style={{ backgroundColor: techColors.primary }} />
          {mainTable && (
            <Tag color="blue" icon={<StarFilled />}>
              Main: {mainTable.name}
            </Tag>
          )}
        </Space>
      }
      style={{ marginBottom: 16 }}
      extra={
        hidden_upload=== true && (
          <Space>
            <EnhancedUpload
              onTablesChange={onTablesChange}
              tables={tables}
            />
          </Space>
        )
      }
    >
      <Card
        size="small"
        type="inner"
        title={
          <Space>
            <TableOutlined />
            <Text strong>All Tables ({allTables.length})</Text>

            {activeTableTab && (
              <Text type="secondary" style={{ fontSize: '12px' }}>

                <Text type="secondary" style={{ fontSize: '10px', marginLeft: '8px' }}>
                  (双击标签取消选中 Active table  )
                </Text>
              </Text>
            )}
          </Space>
        }
        tabList={allTables.map(table => ({
          key: table.id,
          tab: (
            <div
               onClick={(e) => handleTabClick(table.id, e)}
                onDoubleClick={(e) => e.stopPropagation()} // 阻止默认双击行为
                style={{
                  cursor: 'pointer',
                  padding: '8px 12px',
                  margin: '-8px -12px',
                  borderRadius: '4px',
                  backgroundColor: activeTableTab === table.id ? '#e6f7ff' : 'transparent',
                  transition: 'background-color 0.2s'
                }}
              >
            <Space size={4}>
              {/* 修复：使用 isMainTable 函数判断 */}
              {isMainTable(table) ? (
                <>
                  <StarFilled style={{ color: '#faad14', fontSize: '12px' }} />
                  <Tag color="blue" style={{ margin: 0, padding: '0 4px' }}>M</Tag>
                </>
              ) : (
                <Tag color="orange" style={{ margin: 0, padding: '0 4px' }}>A</Tag>
              )}
              <span style={{
                fontWeight: isMainTable(table) ? 'bold' : 'normal',
                color: isMainTable(table) ? techColors.primary : 'inherit'
              }}>
                {table.name}
              </span>
              <Badge
                count={table.columns?.length || 0}
                size="small"
                style={{
                  backgroundColor: isMainTable(table) ? techColors.primary : techColors.orange
                }}
              />
            </Space>
          </div>
          )
        }))}
        activeTabKey={activeTableTab}
        onTabChange={setActiveTableTab}
        style={{ minHeight: '50px' }}
      >
        {allTables.length === 0 ? (
          <div style={{
            textAlign: 'center',
            padding: '60px 20px',
            color: '#999'
          }}>
            <DatabaseOutlined style={{ fontSize: '48px', marginBottom: '16px' }} />
            <div>
              <Text style={{ fontSize: '16px', display: 'block', marginBottom: '8px' }}>
                暂无数据表
              </Text>
              <Text type="secondary">
                请上传数据文件开始处理流程
              </Text>
            </div>
          </div>
        ) : (
          allTables.map(table => (
            <div key={table.id} style={{ display: activeTableTab === table.id ? 'block' : 'none' }}>
              {/* 表头信息 - 同样修复这里的判断 */}

                <Space>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    {isMainTable(table) && (
                      <StarFilled style={{ color: '#faad14', fontSize: '16px' }} />
                    )}
                    <Text strong style={{
                      fontSize: '16px',
                      color: isMainTable(table) ? techColors.primary : 'inherit'
                    }}>
                      {table.name}
                    </Text>
                    {table.originalSheetName && (
                      <Tag color="purple" style={{ fontSize: '10px' }}>
                        {table.originalSheetName}
                      </Tag>
                    )}
                  </div>
                  <Tag color={isMainTable(table) ? 'blue' : 'orange'}>
                    {isMainTable(table) ? '⭐ Main Table' : 'Auxiliary Table'}
                  </Tag>
                  <Tag color="blue">{table.columns?.length || 0} columns</Tag>
                  <Tag color="green">{table.previewData?.length || 0} preview rows</Tag>
                  <Tag color="orange">{table.fullData?.length || 0} total rows</Tag>
                </Space>

                <Space>
                  {/* 修复：只有不是主表的表才显示 Set as Main 按钮 */}
                  {!isMainTable(table) && (
                    <Button
                      size="small"
                      type="primary"
                      ghost
                      icon={<StarOutlined />}
                      onClick={() => onSetMainTable(table.id)}
                    >
                      Set as Main
                    </Button>
                  )}
                  <Button
                    size="small"
                    danger
                    icon={<DeleteOutlined />}
                    onClick={() => onRemoveTable(table.id)}
                  >
                    Remove
                  </Button>

                   {/* 文件信息 - 同样修复这里的判断 */}
              <div style={{ marginBottom: 16 }}>
                <Text strong>Source: </Text>
                  <Text type="secondary" style={{ marginLeft: 8 }}>
                    {table.sourceType === 'url' ? (
                      <Space>
                        <Tag color="green" icon={<LinkOutlined />}>URL</Tag>
                        <Text code style={{ fontSize: '12px', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {table.source}
                        </Text>
                      </Space>
                    ) : (
                      <Space>
                        <Tag color="blue" icon={<UploadOutlined />}>File</Tag>
                        {table.fileName}
                      </Space>
                    )}
                  </Text>

                  {/* 显示数据规模信息 */}
                  {table.totalRows && (
                    <Tag color="purple" style={{ marginLeft: 12 }}>
                      {table.totalRows.toLocaleString()} rows
                    </Tag>
                  )}
                  {table.fileSize && (
                    <Tag color="orange" style={{ marginLeft: 8 }}>
                      {formatFileSize(table.fileSize)}
                    </Tag>
                  )}
              </div>

             </Space>
              {/* 表格预览 */}
              <Collapse
                size="small"
                activeKey={expandedTables.has(table.id) ? ['preview'] : []}
                onChange={() => toggleTableExpansion(table.id)}
              >
                <Panel
                  header={
                    <Space>
                      <EyeOutlined />
                      <span>Data Preview ({table.previewData?.length || 0} rows)</span>
                      <Tag color="cyan">First 10 rows</Tag>
                    </Space>
                  }
                  key="preview"
                >
                  {table.previewData && table.previewData.length > 0 ? (
                    <Table
                      size="small"
                      dataSource={table.previewData.map((item, index) => ({ ...item, key: index }))}
                      columns={(table.columns || []).map(col => ({
                        title: (
                          <Tooltip title={col}>
                            <Text strong style={{ fontSize: '12px' }}>
                              {col.length > 15 ? `${col.substring(0, 15)}...` : col}
                            </Text>
                          </Tooltip>
                        ),
                        dataIndex: col,
                        key: col,
                        width: 120,
                        render: (text) => (
                          <Tooltip title={String(text)}>
                            <Text style={{ fontSize: '11px' }}>
                              {String(text).length > 20 ? `${String(text).substring(0, 20)}...` : String(text)}
                            </Text>
                          </Tooltip>
                        )
                      }))}
                      pagination={false}
                      scroll={{ x: 'max-content', y: 200 }}
                      style={{ background: 'white' }}
                    />
                  ) : (
                    <Alert
                      message="暂无预览数据"
                      description="该表没有可显示的数据"
                      type="info"
                      showIcon
                    />
                  )}
                </Panel>
              </Collapse>

              {/* 列信息 */}
              <div style={{ marginTop: 16 }}>
                <Text strong style={{ fontSize: '14px' }}>Columns ({table.columns?.length || 0}): </Text>
                <div style={{ marginTop: 8, maxHeight: '120px', overflowY: 'auto' }}>
                  {table.columns && table.columns.length > 0 ? (
                    <Space size={[4, 4]} wrap>
                      {table.columns.map(column => (
                        <Tag
                          key={column}
                          color="cyan"
                          style={{
                            fontSize: '12px',
                            margin: '2px',
                            padding: '2px 8px',
                            border: '1px solid #13c2c2'
                          }}
                        >
                          {column}
                        </Tag>
                      ))}
                    </Space>
                  ) : (
                    <Text type="secondary">无列信息</Text>
                  )}
                </div>
              </div>
            </div>
          ))
        )}
      </Card>
    </Card>
  );
};

const SaveConfigModal = ({ visible, onClose, onSave, loading }) => {
  const [configName, setConfigName] = useState('');
  const [description, setDescription] = useState('');

  const handleOk = () => {
    if (!configName.trim()) {
      message.warning('请输入配置名称');
      return;
    }
    onSave(configName.trim(), description.trim());
  };

  const handleCancel = () => {
    setConfigName('');
    setDescription('');
    onClose();
  };

  return (
    <Modal
      title="保存配置"
      open={visible}
      onOk={handleOk}
      onCancel={handleCancel}
      confirmLoading={loading}
      okText="保存"
      cancelText="取消"
    >
      <div style={{ marginBottom: 16 }}>
        <div style={{ marginBottom: 8 }}>
          <strong>配置名称 <span style={{ color: 'red' }}>*</span></strong>
        </div>
        <Input
          placeholder="请输入配置名称"
          value={configName}
          onChange={(e) => setConfigName(e.target.value)}
          onPressEnter={handleOk}
        />
      </div>
      <div>
        <div style={{ marginBottom: 8 }}>
          <strong>配置描述</strong>
        </div>
        <Input.TextArea
          placeholder="请输入配置描述（可选）"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          rows={3}
        />
      </div>
    </Modal>
  );
};


const TableDataPipeline = ({
  form,
  tables: initialTables = [],
  customOperators = [],
  onColumnsChange,
  onTablesChange,
  onchange,
}) => {


  const [processingSteps, setProcessingSteps] = useState([]);
  const [tables, setTables] = useState(initialTables);
  const [savedConfigs, setSavedConfigs] = useState([]);
  const [selectedConfig, setSelectedConfig] = useState(null);
  const [stepExecutions, setStepExecutions] = useState({});
  const [saveModalVisible, setSaveModalVisible] = useState(false);
  const [saveLoading, setSaveLoading] = useState(false);
  const [outputConfig, setOutputConfig] = useState({
    format: 'csv',
    includeSteps: false
  });
  const [configList, setConfigList] = useState([]);
  const [loadingConfigs, setLoadingConfigs] = useState(false);
  const [configManageVisible, setConfigManageVisible] = useState(false);

  // 获取主表
  const mainTable = tables.find(t => t.type === 'main') || tables[0];
  const auxiliaryTables = tables.filter(t => t.type === 'auxiliary' || t !== mainTable);

 console.info("TableDataPipeline --tables:",initialTables)

 useEffect(() => {
    // 获取 form 中的 pipelineSteps 值
    const formPipelineSteps = form?.getFieldValue?.('pipelineSteps');

    if (formPipelineSteps && Array.isArray(formPipelineSteps) && formPipelineSteps.length > 0) {
      console.log("从 form 初始化 processingSteps:", formPipelineSteps);
      setProcessingSteps(formPipelineSteps);
    }
  }, [form]);

 useEffect(() => {
    if (initialTables && initialTables.length > 0) {
      console.log("同步外部 tables:", initialTables);
      setTables(initialTables);
    }
  }, [initialTables]);

  const baseOperators = [
    { value: 'Rename', label: '字段重命名', description: '修改字段名称', icon: <EditOutlined /> },
    { value: 'Sample', label: '数据采样', description: '对数据进行采样', icon: <RocketOutlined /> },
    { value: 'Deduplicate', label: '数据去重', description: '去除重复数据', icon: <DeleteOutlined /> },
    { value: 'Type', label: '类型转换', description: '转换字段数据类型', icon: <CodeOutlined /> },
    { value: 'Filter', label: '数据筛选', description: '根据条件筛选数据', icon: <FilterOutlined /> },
    { value: 'Sort', label: '数据排序', description: '对数据进行排序', icon: <SortAscendingOutlined /> },
    { value: 'DataEdit', label: '数据编辑及新增', description: '创建新的特征字段', icon: <PlusOutlined /> },
    { value: 'Merge', label: '数据合并', description: '两表数据拼接', icon: <ForkOutlined /> },
  ];

  const availableOperators = [...baseOperators, ...customOperators];
  const openSaveModal = () => {
    if (processingSteps.length === 0) {
      message.warning('请先添加处理步骤');
      return;
    }
    setSaveModalVisible(true);
  };
  // 表管理相关函数
  const handleTablesChange = (newTables) => {
    setTables(newTables);
    onTablesChange?.(newTables);
  };


  const handleExportJson = () => {
    const exportData = {
      pipelineConfig: {
        mainTable: mainTable?.name,
        inputColumns: availableColumns,
        totalSteps: processingSteps.length,
        outputConfig: outputConfig,
        pipelineSteps: processingSteps
      }
    };
    const dataStr = JSON.stringify(exportData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `pipeline-config-${new Date().getTime()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    message.success('配置已导出为JSON文件');
};

  const handleAddTable = () => {
    const newTable = {
      id: `table_${Date.now()}`,
      name: `Auxiliary_${auxiliaryTables?.length + 1}`,
      fileName: 'manual_add.csv',
      type: 'auxiliary',
      columns: ['id', 'feature1', 'feature2'],
      previewData: Array.from({ length: 5 }, (_, i) => ({
        id: i + 1,
        feature1: `value_${i + 1}`,
        feature2: Math.random().toFixed(4)
      }))
    };
    handleTablesChange([...tables, newTable]);
  };

  const handleRemoveTable = (tableId) => {
    const newTables = tables.filter(t => t.id !== tableId);
    handleTablesChange(newTables);
  };

  const handleSetMainTable = (tableId) => {
    const newTables = tables.map(table => ({
      ...table,
      type: table.id === tableId ? 'main' : 'auxiliary'
    }));
    handleTablesChange(newTables);
  };

const executeAllSteps = async () => {
  if (processingSteps.length === 0) {
    message.warning('请先添加处理步骤');
    return;
  }

  if (!mainTable) {
    message.error('请先上传主表数据');
    return;
  }

  const loadingStates = {};
  processingSteps.forEach(step => {
    loadingStates[step.id] = { loading: true };
  });
  setStepExecutions(loadingStates);
  const requestData = {
      pipelineId: `pipeline_${Date.now()}`,
      mainTable: {
        name: mainTable.name, // data: mainTable.previewData, 或者 mainTable.fullData 如果是完整数据
        columns: mainTable.columns,
        fileName: mainTable.fileName,
        filehash: mainTable.filehash
      },
      auxiliaryTables: auxiliaryTables.map(table => ({
        name: table.name,    //data: table.previewData, 或者 table.fullData

        columns: table.columns,
        fileName: table.fileName,
        filehash: mainTable.filehash
      })),
      processingSteps: processingSteps.map(step => ({
        id: step.id,
        type: step.type,
        config: step.config,
        index: processingSteps.indexOf(step)
      })),
      outputConfig: outputConfig
    };

   if(form)
   {
     onchange( { tables, pipelineSteps:processingSteps  }  );
     console.info("trigger onchange to save data");
     setStepExecutions( [] )
     return
   }
       else{

      try {


        console.log('发送给后端的配置:', requestData);

        // 调用后端API
        const response = await axios.post('/api/tablepipeline/execute-pipeline', JSON.stringify(requestData), {withCredentials:true});

        console.log('后端返回结果:', response.data);

        if (response.data.status === 'success') {
          const resultData = response.data.data;
          const stepResults = {};

          // 处理步骤执行结果
          if (resultData.result_data.step_results && resultData.result_data.step_results.length > 0) {
            resultData.result_data.step_results.forEach((stepResult, index) => {
              const stepId = processingSteps[index]?.id;
              if (stepId) {
                stepResults[stepId] = {
                  success: stepResult.success,
                  result: stepResult.output_sample_data,
                  message: stepResult.message,
                  outputColumns: stepResult.output_columns || stepResult.outputColumns || [],
                  dataLength: stepResult.output_data_length || stepResult.dataLength || 0,
                  rowsChanged: stepResult.rows_changed || 0 ,
                  executionTime: stepResult.total_processing_time||   stepResult.total_processing_time || stepResult.executionTime,
                  // 步骤特定的详细信息
                  stepDetails: stepResult
                };
              }
            });
          }
          console.info("stepResults:",stepResults)

          // 处理最终输出
          if (resultData.final_output) {
            const finalOutput = resultData.result_data;
            stepResults.final = {
              success: true,
              result: resultData.result_data.final_sample_data || [],
              message: `处理完成，共 ${finalOutput?.final_count || 0} 行数据`,
              outputColumns: finalOutput.final_columns || [],
              dataLength: finalOutput.data_length || 0,
              outputFile: finalOutput.output_file,
              executionTime: finalOutput.processing_summary?.total_processing_time || 'unknown'
            };

            // 更新最终的列信息
            onColumnsChange?.(finalOutput.columns || []);
          }

          // 如果有 result_data，也处理一下
          if (resultData.result_data) {
            const resultDataObj = resultData.result_data;
            stepResults.summary = {
              success: resultDataObj.success,
              message: `流水线执行完成: ${resultDataObj.completed_steps}/${resultDataObj.total_steps} 个步骤成功`,
              originalCount: resultDataObj.original_count,
              finalCount: resultDataObj.final_count,
              outputFile: resultDataObj.output_file,
              stepResults: resultDataObj.step_results || []
            };
          }

          setStepExecutions(stepResults);
          message.success(`流水线执行完成: ${resultData.result_data?.completed_steps || 0}/${resultData.result_data?.total_steps || 0} 个步骤成功`);

        } else {
          throw new Error(response.data.message || '后端执行失败');
        }

      } catch (error) {
        console.error('执行失败:', error);
        message.error(`执行失败: ${error.message}`);

        // 设置所有步骤为失败状态
        const errorStates = {};
        processingSteps.forEach(step => {
          errorStates[step.id] = {
            success: false,
            message: error.message,
            loading: false
          };
        });
        setStepExecutions(errorStates);
      }
  }


};

  const ExecutionResults = ({ stepExecutions, processingSteps, mainTable }) => {
  const [activeResultTab, setActiveResultTab] = useState('final');

  if (Object.keys(stepExecutions).length === 0) {
    return null;
  }

  const hasFinalResult = stepExecutions.final;
  const hasStepResults = processingSteps.some(step => stepExecutions[step.id]);
  const hasSummary = stepExecutions.summary;

  const resultTabs = [];

  if (hasFinalResult) {
    resultTabs.push({ key: 'final', tab: '最终结果', result: stepExecutions.final });
  }

  if (hasSummary) {
    resultTabs.push({ key: 'summary', tab: '执行摘要', result: stepExecutions.summary });
  }

  processingSteps.forEach((step, index) => {
    const stepResult = stepExecutions[step.id];
    if (stepResult) {
      resultTabs.push({
        key: step.id,
        tab: `步骤 ${index + 1}`,
        result: stepResult,
        step
      });
    }
  });

  const renderResultContent = (result, step) => {
    if (!result) return null;
    console.info("result:",result)
    return (
      <div style={{ padding: '16px' }}>
        {/* 结果状态头 */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '16px',
          padding: '12px',
          background: result.success ? '#f6ffed' : '#fff2f0',
          border: `1px solid ${result.success ? '#b7eb8f' : '#ffccc7'}`,
          borderRadius: '6px'
        }}>
          <div>
            <Text strong style={{ color: result.success ? '#52c41a' : '#ff4d4f' }}>
              {result.success ? '✓ 执行成功' : '✗ 执行失败'}
            </Text>
            {result.message && (
              <div style={{ marginTop: '4px' }}>
                <Text type="secondary">{result.message}</Text>
              </div>
            )}
          </div>
          <Space>
            {result.executionTime && (
              <Tag color="blue">耗时: {result.executionTime}</Tag>
            )}
            {result.dataLength > 0 && (
              <Tag color="green">{result.dataLength} 行数据</Tag>
            )}
          </Space>
        </div>

        {/* 数据预览 */}
        {result && result.outputColumns.length > 0 && (
          <Card
            size="small"
            title="数据预览"
            style={{ marginBottom: '16px' }}
            extra={
              <Space>
                <Tag color="cyan">{result.outputColumns?.length || 0} 列</Tag>
                <Tag color="orange">{result.result.length} 行样例</Tag>
              </Space>
            }
          >
            <Table
              size="small"
              dataSource={result.result.map((item, i) => ({ ...item, key: i }))}
              columns={(result.outputColumns || Object.keys(result.result[0] || {})).map(col => ({
                title: (
                  <Tooltip title={col}>
                    <Text strong style={{ fontSize: '12px' }}>
                      {col}
                    </Text>
                  </Tooltip>
                ),
                dataIndex: col,
                key: col,
                width: 120,
                render: (text) => (
                  <Tooltip title={String(text)}>
                    <Text style={{ fontSize: '11px' }}>
                      {String(text).length > 15 ? `${String(text).substring(0, 15)}...` : String(text)}
                    </Text>
                  </Tooltip>
                )
              }))}
              pagination={false}
              scroll={{ y: 200 }}
            />
          </Card>
        )}

        {/* 步骤特定信息 */}
        {step && step.type === 'rename' && result.stepDetails && (
          <Card size="small" title="重命名详情" style={{ marginBottom: '16px' }}>
            <Space wrap>
              {result.stepDetails.rename_mapping && Object.entries(result.stepDetails.rename_mapping).map(([oldName, newName]) => (
                <Tag key={oldName} color="blue">
                  {oldName} → {newName}
                </Tag>
              ))}
              {result.stepDetails.renamed_columns && (
                <Tag color="green">重命名了 {result.stepDetails.renamed_columns} 个字段</Tag>
              )}
            </Space>
          </Card>
        )}

        {/* 输出文件信息 */}
        {result.outputFile && (
              <Card size="small" title="输出文件">
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '8px 0'
                }}>
                  {/* 文件信息展示 */}
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    flex: 1
                  }}>
                    <FileTextOutlined style={{ color: '#1890ff', fontSize: '16px' }} />
                    <div>
                      <div style={{
                        fontWeight: 500,
                        marginBottom: '4px',
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis'
                      }}>
                        {result.outputFile.split('/').pop()}
                      </div>
                      <div style={{
                        fontSize: '12px',
                        color: '#666',
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis'
                      }}>
                        {result.outputFile}
                      </div>
                    </div>
                  </div>

                  {/* 下载按钮 */}
                  <Button
                    type="primary"
                    icon={<DownloadOutlined />}
                    size="small"
                    onClick={() => {
                      const downloadUrl = getFileUrl(result.outputFile);
                      const link = document.createElement('a');
                      link.href = downloadUrl;
                      const fileName = result.outputFile.split('/').pop();
                      link.download = fileName;
                      document.body.appendChild(link);
                      link.click();
                      document.body.removeChild(link);
                    }}
                  >
                    下载
                  </Button>
                </div>
              </Card>
            )}

      </div>
    );
  };

  return (
    <Card
      title={
        <Space>
          <PlayCircleOutlined style={{ color: techColors.success }} />
          <span>执行结果</span>
          <Badge
            count={resultTabs.length}
            style={{ backgroundColor: techColors.success }}
          />
        </Space>
      }
      style={{
        marginTop: '16px',
        background: 'white',
        boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
      }}
      tabList={resultTabs.map(tab => ({
        key: tab.key,
        tab: (
          <Space size={4}>
            {stepExecutions[tab.key]?.success ? (
              <span style={{ color: techColors.success }}>✓</span>
            ) : (
              <span style={{ color: techColors.error }}>✗</span>
            )}
            {tab.tab}
          </Space>
        )
      }))}
      activeTabKey={activeResultTab}
      onTabChange={setActiveResultTab}
    >
      {renderResultContent(
        resultTabs.find(tab => tab.key === activeResultTab)?.result,
        resultTabs.find(tab => tab.key === activeResultTab)?.step
      )}
    </Card>
  );
};

  // 计算列信息
  const computeColumnsAfterStep = (step, inputColumns) => {
    let outputColumns = [...inputColumns];

    if (step.type === 'rename' && step.config.renameMapping) {
      outputColumns = outputColumns.map(col => {
        const mapped = step.config.renameMapping[col];
        return mapped !== undefined ? mapped : col;
      });
    } else if (step.type === 'feature_derive' && step.config.newFeatures) {
      outputColumns = [...outputColumns, ...step.config.newFeatures];
    }

    return outputColumns;
  };

  const computeColumnsAfterSteps = (steps, startColumns = mainTable?.columns || []) => {
    let columns = [...startColumns];
    steps.forEach(step => {
      columns = computeColumnsAfterStep(step, columns);
    });
    return columns;
  };

  // 步骤管理函数
  const addProcessingStep = (operatorType) => {
    const newStep = {
      id: `step_${Date.now()}`,
      type: operatorType,
      config: getDefaultConfig(operatorType)
    };
    const newSteps = [...processingSteps, newStep];
    setProcessingSteps(newSteps);
    updateFormConfig(newSteps);
  };

  const removeProcessingStep = (index) => {
    const newSteps = processingSteps.filter((_, i) => i !== index);
    setProcessingSteps(newSteps);
    updateFormConfig(newSteps);
  };

  const moveStep = (index, direction) => {
    if ((direction === 'up' && index === 0) || (direction === 'down' && index === processingSteps.length - 1)) {
      return;
    }

    const newSteps = [...processingSteps];
    const newIndex = direction === 'up' ? index - 1 : index + 1;
    [newSteps[index], newSteps[newIndex]] = [newSteps[newIndex], newSteps[index]];
    setProcessingSteps(newSteps);
    updateFormConfig(newSteps);
  };

  const updateStepConfig = (index, newConfig) => {
    const newSteps = [...processingSteps];
    newSteps[index].config = { ...newSteps[index].config, ...newConfig };
    setProcessingSteps(newSteps);
    updateFormConfig(newSteps);
    console.info("newSteps:",newSteps);
  };

  const updateFormConfig = (steps) => {
    const updatedColumns = computeColumnsAfterSteps(steps);
    if (form) {
      form.setFieldsValue({
        pipelineSteps: steps,
        outputColumns: updatedColumns,
        tables:tables,
      });
    }
    onColumnsChange?.(updatedColumns);
  };

  const getDefaultConfig = (operatorType) => {
    const defaults = {
      rename: { renameMapping: {} },
      sample: {
        sampleType: 'whole',
        sampleMethod: 'downsample',
        sampleRatio: 10,
        sampleColumn: '',
        sampleCondition: '',
        sampleConditionValue: ''
      },
      deduplicate: { deduplicateColumns: [], keepMethod: 'first' },
      typeTransform: { transformConfigs: [] },
      filter: { conditions: [] },
      sort: { sortBy: [] },
      feature_derive: { newFeatures: [] }
    };
    return defaults[operatorType] || {};
  };

  // 配置管理
   const handleSaveConfig = async (configName, description) => {
    if (!tables || tables.length === 0) {
      message.warning('请先上传数据表');
      return;
    }

    if (processingSteps.length === 0) {
      message.warning('请至少添加一个处理步骤');
      return;
    }

    const configData = {
      name: configName,
      description: description,
      pipelineSteps: processingSteps,
      outputConfig: outputConfig
    };

    console.info("准备保存的配置:", configData);
    setSaveLoading(true);

    try {
      const response = await axios.post('/api/tablepipeline/save-config', configData, { withCredentials: true });

      if (response.data.success) {
        message.success('配置保存成功');
        setSaveModalVisible(false);

        // 可选：更新本地保存的配置列表
        setSavedConfigs(prev => [...prev, {
          ...configData,
          id: response.data.id || configData.id
        }]);
      } else {
        throw new Error(response.data.message || '保存失败');
      }
    } catch (error) {
      console.error('保存配置失败:', error);
      message.error(`保存配置失败: ${error.response?.data?.message || error.message}`);
    } finally {
      setSaveLoading(false);
    }
  };

  const loadConfigList = async () => {
    setLoadingConfigs(true);
    try {
      const response = await axios.get('/api/tablepipeline/configs', { withCredentials: true } );

      if (response.data.success) {
        setConfigList(response.data.data || []);
      } else {
        throw new Error(response.data.message || '加载配置列表失败');
      }
    } catch (error) {
      console.error('加载配置列表失败:', error);
      message.error(`加载配置列表失败: ${error.response?.data?.message || error.message}`);
    } finally {
      setLoadingConfigs(false);
    }
  };

  const handleLoadConfig = async (configId) => {
    try {
      const response = await axios.get(`/api/tablepipeline/config/${configId}`, { withCredentials:true });
      if (response.data.success) {
        const config = response.data.data;

        // 设置配置到状态
        setProcessingSteps(config.pipelineSteps || []);
        setTables(config.tables || []);
        setOutputConfig(config.outputConfig || { format: 'csv', includeSteps: false });
        setSelectedConfig(configId);

        message.success(`配置 "${config.name}" 加载成功`);

        // 关闭管理模态框（如果打开的话）
        setConfigManageVisible(false);
      } else {
        throw new Error(response.data.message || '加载配置失败');
      }
    } catch (error) {
      console.error('加载配置失败:', error);
      message.error(`加载配置失败: ${error.response?.data?.message || error.message}`);
    }
  };

  // 删除配置
  const handleDeleteConfig = async (configId, configName) => {
    try {
      const response = await axios.delete(`/api/tablepiple/config/${configId}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      if (response.data.success) {
        message.success(`配置 "${configName}" 删除成功`);
        // 重新加载配置列表
        loadConfigList();
        // 如果删除的是当前选中的配置，清空选中状态
        if (selectedConfig === configId) {
          setSelectedConfig(null);
        }
      } else {
        throw new Error(response.data.message || '删除配置失败');
      }
    } catch (error) {
      console.error('删除配置失败:', error);
      message.error(`删除配置失败: ${error.response?.data?.message || error.message}`);
    }
  };

  // 配置管理模态框
  const ConfigManageModal = () => (
    <Modal
      title="配置管理"
      open={configManageVisible}
      onCancel={() => setConfigManageVisible(false)}
      footer={null}
      width={700}
    >
      <div style={{ marginBottom: 16, textAlign: 'right' }}>
        <Button
          icon={<ReloadOutlined />}
          onClick={loadConfigList}
          loading={loadingConfigs}
        >
          刷新列表
        </Button>
      </div>

      <List
        loading={loadingConfigs}
        dataSource={configList}
        renderItem={(config) => (
          <List.Item
            actions={[
              <Button
                type="link"
                onClick={() => handleLoadConfig(config.id)}
                disabled={!config.steps || config.steps === 0}
              >
                加载
              </Button>,
              <Button
                type="link"
                danger
                onClick={() => {
                  Modal.confirm({
                    title: '确认删除',
                    content: `确定要删除配置 "${config.name}" 吗？`,
                    onOk: () => handleDeleteConfig(config.id, config.name)
                  });
                }}
              >
                删除
              </Button>
            ]}
          >
            <List.Item.Meta
              title={
                <Space>
                  <span>{config.name}</span>
                  {selectedConfig === config.id && <Tag color="blue">当前加载</Tag>}
                </Space>
              }
              description={
                <Space direction="vertical" size={0}>
                  <div>
                    <Tag color="blue">{config.steps || 0} 个步骤</Tag>
                    <Tag color="green">{config.tables || 0} 个表</Tag>
                  </div>
                  <div style={{ color: '#999', fontSize: '12px' }}>
                    创建时间: {config.created}
                    {config.description && ` | 描述: ${config.description}`}
                  </div>
                </Space>
              }
            />
          </List.Item>
        )}
        locale={{ emptyText: '暂无保存的配置' }}
      />
    </Modal>
  );

//    case 'append':
//      return <AppendConfig {...commonProps} />;
  // 渲染步骤配置
 const renderStepConfig = (step, index, availableColumns,maintable_name="", auxiliary_list=[],inputData=[]) => {
  const commonProps = {
    config: step.config,
    preColumns: availableColumns,
    onChange: (newConfig) => updateStepConfig(index, newConfig)
  };

  switch (step.type) {
    case 'Rename':
      return <RenameConfig config = {step.config}
                           preColumns={availableColumns}
                           onChange = {(config) => updateStepConfig(index,config)} />;
    case 'Sample':
      return <SampleConfig config = {step.config}
                           preColumns={availableColumns}
                           onChange = {(config) => updateStepConfig(index,config)} />;
    case 'Deduplicate':
      return <DeduplicateConfig config = {step.config}
                           preColumns={availableColumns}
                           onChange = {(config) => updateStepConfig(index,config)} />;
    case 'Filter':
      return <FilterDataConfig config = {step.config}
                           dataColumns={availableColumns}
                           onChange = {(config) => updateStepConfig(index,config)} />;
    case 'Sort':
      return <SortConfig config = {step.config}
                           preColumns={availableColumns}
                           onChange = {(config) => updateStepConfig(index,config)} />;
    case 'Type':
    case 'TypeConfig':
      return <TableTypeConfig
              dataColumns={availableColumns}
              config     ={step.config}
              onChange   ={(config) => updateStepConfig(index,config)}
            />;
    case 'Merge':
          const allNodeColumns = [
            {
              id: maintable_name,
              label: maintable_name,
              columns: availableColumns
            },
            ...auxiliary_list.map(tab => ({
              id: tab.id,
              label: tab.label,
              columns: tab.columns
            }))
          ];

          return (
                <MergeDataConfig
                  connectedNodes={allNodeColumns}
                  handleonchange={(newConfig) => {
                    console.log('Merge配置更新:', newConfig);
                    updateStepConfig(index, newConfig);
                  }}
                />
          );

    // 高级数据处理算子
    case 'DataEdit':
      return (
        <DataEditOperator
          form={form} // 需要从父组件传递form实例
          inputData={inputData} // 需要从父组件传递当前数据
          rowCount={availableColumns?.length || 0}
          {...commonProps}
        />
      );


    // 默认情况
    default:
      const customOperator = customOperators.find(op => op.value === step.type);
      if (customOperator && customOperator.component) {
        return React.createElement(customOperator.component, commonProps);
      }
      return (
        <Alert
          message={`未知操作类型: ${step.type}`}
          type="warning"
          showIcon
        />
      );
  }
};

  const availableColumns = mainTable?.columns || [];
  const currentColumns = computeColumnsAfterSteps(processingSteps);

  return (
    <div style={{
      padding: 24,
      background: 'linear-gradient(135deg, #f5f7ff 0%, #f0f2ff 100%)',
      minHeight: '50vh'
    }}>
      {/* 头部标题和统计 */}
      <Card
      style={{
        marginBottom: 4,
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        color: 'white',
        border: 'none',
        position: 'relative',
        overflow: 'hidden'
      }}
      bodyStyle={{ padding: '16px 20px', position: 'relative' }}
    >

        <Row align="middle" gutter={14}>
          <Col>
            <RocketOutlined style={{ fontSize: 24, color: 'white' }} />
          </Col>
          <Col flex="1">
            <Title level={4} style={{ color: 'white', margin: 0, fontSize: '24px' }}>
              Tabledata Pipeline
            </Title>
            <div style={{ marginTop: '1px' }}>
          <Space size={4}>
            <StarFilled style={{ color: '#ffd666', fontSize: '12px' }} />
            <Text style={{
              color: 'rgba(255,255,255,0.9)',
              fontSize: '14px',
              fontWeight: '500'
            }}>
              Excel++: Configure once, reuse forever • Automated processing • Scale to big data
            </Text>
            <StarFilled style={{ color: '#ffd666', fontSize: '12px' }} />
          </Space>

      </div>
          </Col>
          <Col>
            <Space>
              <Button
                icon={<ExportOutlined />}
                onClick={openSaveModal}
                style={{
                  background: 'rgba(255,255,255,0.2)',
                  border: '1px solid rgba(255,255,255,0.3)',
                  color: 'white'
                }}
              >
                Save Config
              </Button>

              <Select
                style={{ width: 200 }}
                placeholder="Load saved config"
                value={selectedConfig}
                onChange={handleLoadConfig}
                loading={loadingConfigs}
                dropdownRender={(menu) => (
                    <div>
                      {menu}
                      <div style={{ padding: '8px 12px', borderTop: '1px solid #f0f0f0' }}>
                        <Button
                          type="link"
                          icon={<SettingOutlined />}
                          onClick={() => setConfigManageVisible(true)}
                          style={{ padding: 0 }}
                        >
                          管理配置
                        </Button>
                      </div>
                    </div>
                  )}
               >
              {configList.map(config => (
                <Option key={config.id} value={config.id} disabled={!config.pipelineSteps || config.pipelineSteps.length === 0}>
                  <div>
                    <div>{config.name}</div>
                    <div style={{ fontSize: '12px', color: '#999' }}>
                      步骤: {config.steps || 0} | 表: {config.tables || 0}
                    </div>
                  </div>
                </Option>
              ))}

              </Select>

              <Badge count={tables.length} showZero style={{ backgroundColor: '#52c41a' }}>
                <Tag color="blue">Tables</Tag>
              </Badge>
              <Badge count={availableColumns.length} showZero style={{ backgroundColor: '#52c41a' }}>
                <Tag color="blue">Fields</Tag>
              </Badge>
              <Badge count={processingSteps.length} showZero style={{ backgroundColor: '#faad14' }}>
                <Tag color="orange">Steps</Tag>
              </Badge>

              <ConfigManageModal />
            </Space>
          </Col>
        </Row>
      </Card>
      {/* 保存配置模态框 */}
      <SaveConfigModal
        visible={saveModalVisible}
        onClose={() => setSaveModalVisible(false)}
        onSave={handleSaveConfig}
        loading={saveLoading}
      />
      {/* 表管理 */}
      <TableManager
        tables={tables}
        onTablesChange={handleTablesChange}
        onAddTable={handleAddTable}
        onRemoveTable={handleRemoveTable}
        onSetMainTable={handleSetMainTable}
        hidden_upload ={ form !==null  }
      />

      {/* 主表数据预览 */}
      {mainTable && (
        <DataPreview
          data={mainTable.previewData}
          columns={mainTable.columns}
          fileName={mainTable.fileName}
          title="Main Table Preview"
        />
      )}

      {/* 操作选择器和输出配置 */}
      <Card
        style={{
          marginBottom: 16,
          background: 'white',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
        }}
      >
        <Row gutter={16} align="middle">
          <Col span={12}>
            <div>
              <Text strong style={{ fontSize: '16px' }}>添加处理步骤: </Text>
              <Space.Compact style={{ marginLeft: 12, verticalAlign: 'top' }}>
                  {/* 精简下拉选择器 */}
                  <Select
                    style={{ width: 180 }}
                    placeholder="选择操作..."
                    onChange={addProcessingStep}
                    disabled={availableColumns.length === 0}
                    dropdownStyle={{ minWidth: 220 }}
                    optionLabelProp="label"
                  >
                    {availableOperators.map(op => (
                      <Select.Option key={op.value} value={op.value} label={op.label}>
                        <Space>
                          <span style={{ color: techColors.primary }}>{op.icon}</span>
                          <div>
                            <div style={{ fontWeight: 'bold' }}>{op.label}</div>
                            <Text type="secondary" style={{ fontSize: 12 }}>
                              {op.description}
                            </Text>
                          </div>
                        </Space>
                      </Select.Option>
                    ))}
                  </Select>
                  <Space size={[4, 8]} style={{ marginLeft: 28 }}>
                      {availableOperators.map(op => ( // 只显示前4个常用功能
                        <Tooltip key={op.value} title={op.description}>
                          <Button
                            icon={op.icon}
                            size="small"
                            onClick={() => addProcessingStep(op.value)}
                            disabled={availableColumns.length === 0}
                            style={{
                              width: '32px',
                              height: '32px',
                              padding: 0,
                              background: 'white',
                              border: `1px solid ${techColors.primary}30`,
                              color: techColors.primary,
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center'
                            }}
                            onMouseEnter={(e) => {
                              e.currentTarget.style.background = '#f0f8ff';
                              e.currentTarget.style.borderColor = techColors.primary;
                            }}
                            onMouseLeave={(e) => {
                              e.currentTarget.style.background = 'white';
                              e.currentTarget.style.borderColor = `${techColors.primary}30`;
                            }}
                          />
                        </Tooltip>
                      ))}
                  </Space>
              </Space.Compact>
            </div>
          </Col>


          <Col span={12}>
            <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: 12 }}>
              <Text strong>输出配置:</Text>
              <Radio.Group
                value={outputConfig.format}
                onChange={(e) => setOutputConfig({...outputConfig, format: e.target.value})}
              >
                <Radio.Button value="csv">CSV</Radio.Button>
                <Radio.Button value="excel">Excel</Radio.Button>
              </Radio.Group>

              <Switch
                checked={outputConfig.includeSteps}
                onChange={(checked) => setOutputConfig({...outputConfig, includeSteps: checked})}
              />
              <Text>包含步骤数据</Text>

              <Button
                type="primary"
                icon={<PlayCircleOutlined />}
                size="large"
                onClick={executeAllSteps}
                loading={Object.values(stepExecutions).some(execution => execution?.loading)}
                style={{
                  background: `linear-gradient(135deg, ${techColors.success}, ${techColors.info})`,
                  border: 'none'
                }}
              >
                { form===null?`执行全部步骤`:`保存全部步骤` }
              </Button>
            </div>
          </Col>
        </Row>

        {availableColumns.length === 0 && (
          <Alert
            message="暂无可用字段"
            description="请先上传数据文件或确保数据已正确加载"
            type="warning"
            showIcon
            style={{ marginTop: 12 }}
          />
        )}
      </Card>

      {/* 处理步骤列表 */}
      {processingSteps.length > 0 ? (
        <Card
          title={
            <Space>
              <RocketOutlined style={{ color: techColors.primary }} />
              <span>Processing Pipeline</span>
              <Tag color="blue">{processingSteps.length} Steps</Tag>
              <Text type="secondary" style={{ fontSize: '12px' }}>
                Main Table: {availableColumns.length} columns, {mainTable?.previewData?.length || 0} rows
              </Text>
            </Space>
          }
          style={{
            marginBottom: 16,
            background: 'white',
            boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
            borderRadius: '8px'
          }}
        >
          <div style={{ maxHeight: '600px', overflow: 'auto' }}>
            {processingSteps.map((step, index) => {
              const operatorInfo = availableOperators.find(op => op.value === step.type);
              const inputColumns = computeColumnsAfterSteps(processingSteps.slice(0, index));
              const outputColumns = computeColumnsAfterSteps(processingSteps.slice(0, index + 1));
              const executionResult = stepExecutions[step.id];

              return (
                <div
                  key={step.id}
                  style={{
                    padding: '20px',
                    marginBottom: '16px',
                    border: '2px solid #f0f0f0',
                    borderRadius: '12px',
                    background: 'linear-gradient(135deg, #fafafa 0%, #f8f9fa 100%)',
                    position: 'relative'
                  }}
                >
                  {/* 步骤头部 */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <div
                        style={{
                          width: '44px',
                          height: '44px',
                          borderRadius: '12px',
                          background: `linear-gradient(135deg, ${techColors.primary}, ${techColors.info})`,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          color: 'white',
                          fontWeight: 'bold',
                          fontSize: '16px'
                        }}
                      >
                        {index + 1}
                      </div>
                      <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                          <span style={{ color: techColors.primary, fontSize: '18px' }}>
                            {operatorInfo?.icon}
                          </span>
                          <Text strong style={{ fontSize: '16px' }}>{operatorInfo?.label}</Text>
                          {executionResult && (
                            <Tag color={executionResult.success ? 'success' : 'error'} style={{ borderRadius: '12px' }}>
                              {executionResult.success ? 'Executed' : 'Failed'}
                            </Tag>
                          )}
                        </div>
                        <Space size="small">
                          <Tag color="blue" style={{ borderRadius: '6px', fontSize: '12px' }}>
                            Input: {inputColumns.length} cols
                          </Tag>
                          <Tag color="green" style={{ borderRadius: '6px', fontSize: '12px' }}>
                            Output: {outputColumns.length} cols
                          </Tag>
                        </Space>
                      </div>
                    </div>

                    {/* 操作按钮 */}
                    <Space>

                      <Tooltip title="Move Up">
                        <Button
                          icon={<ArrowUpOutlined />}
                          size="small"
                          disabled={index === 0}
                          onClick={() => moveStep(index, 'up')}
                          style={{ borderRadius: '6px' }}
                        />
                      </Tooltip>
                      <Tooltip title="Move Down">
                        <Button
                          icon={<ArrowDownOutlined />}
                          size="small"
                          disabled={index === processingSteps.length - 1}
                          onClick={() => moveStep(index, 'down')}
                          style={{ borderRadius: '6px' }}
                        />
                      </Tooltip>
                      <Tooltip title="Delete">
                        <Button
                          icon={<DeleteOutlined />}
                          size="small"
                          danger
                          onClick={() => removeProcessingStep(index)}
                          style={{ borderRadius: '6px' }}
                        />
                      </Tooltip>
                    </Space>
                  </div>

                  {/* 步骤内容和执行结果 */}
                  <div style={{ display: 'flex', gap: '16px' }}>

                    <div style={{ flex: 1 }}>
                     <ConfigWrapper
                        step={step}
                        onConfigChange={(newConfig) => updateStepConfig(index, newConfig)}
                        availableColumns={inputColumns}
                      >
                        {renderStepConfig(step, index, inputColumns, mainTable.name,auxiliaryTables.map( tab =>({  id:tab.name,columns:tab.columns }) ) ) }
                      </ConfigWrapper>
                    </div>

                    {/* 执行结果区域 */}
                    {executionResult && (
                      <div style={{
                        flex: 1,
                        border: `2px solid ${executionResult.success ? techColors.success : techColors.error}`,
                        borderRadius: '8px',
                        padding: '12px',
                        background: 'white'
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                          <Text strong>Execution Result</Text>
                          <Tag color={executionResult.success ? 'success' : 'error'}>
                            {executionResult.success ? 'Success' : 'Failed'}
                          </Tag>
                        </div>
                        {executionResult.result && executionResult.result.length > 0 && (
                          <Table
                            size="small"
                            dataSource={executionResult.result.slice(0, 3).map((item, i) => ({ ...item, key: i }))}
                            columns={Object.keys(executionResult.result[0] || {}).map(col => ({
                              title: col,
                              dataIndex: col,
                              key: col,
                              width: 100,
                              render: (text) => (
                                <Text style={{ fontSize: '11px' }}>
                                  {String(text).length > 10 ? `${String(text).substring(0, 10)}...` : String(text)}
                                </Text>
                              )
                            }))}
                            pagination={false}
                            scroll={{ y: 120 }}
                          />
                        )}
                        {executionResult.message && (
                          <Text type="secondary" style={{ fontSize: '12px' }}>
                            {executionResult.message}
                          </Text>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      ) : (
        <Card
          style={{
            textAlign: 'center',
            padding: 60,
            background: 'white',
            boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
          }}
        >
          <RocketOutlined style={{ fontSize: 64, color: '#d9d9d9', marginBottom: 16 }} />
          <Title level={4} type="secondary">开始构建数据处理流水线</Title>
          <Text type="secondary">
            选择上方的处理操作来添加第一个步骤，构建您的数据处理流程
          </Text>
        </Card>
      )}

      <ExecutionResults
      stepExecutions={stepExecutions}
      processingSteps={processingSteps}
      mainTable={mainTable}
     />

      {/* 配置预览 */}
      {processingSteps.length > 0 && (
        <Card
          title={
            <Space>
              <CodeOutlined style={{ color: techColors.purple }} />
              <span>配置总览</span>
            </Space>
          }
          size="small"
          extra={
            <Button
              icon={<ExportOutlined />}
              size="small"
              onClick={handleExportJson}
              type="primary"
              ghost
              style={{ borderColor: techColors.purple, color: techColors.purple }}
            >
              导出配置
            </Button>
          }
          style={{
            background: 'white',
            boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
          }}
        >
          <ReactJson
            src={{
              tables: tables.map(t => ({
                name: t.name,
                type: t.type,
                filehash:t.filehash,
                columns: t.columns,
                rowCount: t.previewData.length
              })),
              mainTable: mainTable?.name,
              inputColumns: availableColumns,
              outputColumns: currentColumns,
              totalSteps: processingSteps.length,
              outputConfig: outputConfig,
              pipelineSteps: processingSteps
            }}
            name="pipelineConfig"
            collapsed={1}
            displayDataTypes={false}
            displayObjectSize={false}
            enableClipboard={false}
            style={{
              padding: '16px',
              background: '#ffffff',
              borderRadius: '6px',
              border: '1px solid #e8e8e8',
              fontSize: '13px'
            }}
            theme="rjv-default"
            iconStyle="circle"
            indentWidth={2}
          />
        </Card>
      )}
    </div>
  );
};

const SortAscendingOutlined = () => <span>↕️</span>;

export default TableDataPipeline;