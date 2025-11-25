import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Form, Input, Select, Button, Table, Space, Card, Row, Col, message, Switch, Typography,
  InputNumber, Divider, Collapse, Tabs, Radio, Alert, Slider, Tag, Tooltip, Popconfirm, Upload, Badge,
  Descriptions, Progress, Statistic, Empty, Result,Modal
} from 'antd';
import axios from 'axios';
import {
  PlusOutlined, DeleteOutlined, EditOutlined, SaveOutlined, GlobalOutlined,
  InfoCircleOutlined, QuestionCircleOutlined, CodeOutlined, ExperimentOutlined,
  ThunderboltOutlined, PlayCircleOutlined, DatabaseOutlined, UploadOutlined,
  ExportOutlined, FileTextOutlined, EyeOutlined, ColumnHeightOutlined,
  SettingOutlined, CheckCircleOutlined, ExclamationCircleOutlined,
  BookOutlined, LineChartOutlined, SafetyOutlined, LockOutlined,BarChartOutlined,StarFilled,
  RocketOutlined, ApiOutlined,SearchOutlined,
} from '@ant-design/icons';
import ReactJson from '@microlink/react-json-view';
import { i18n, ALGORITHM_CONFIGS, MODEL_CONFIGS, SCENARIO_TEMPLATES  } from './i18n';

const { Option, OptGroup } = Select;
const { Text, Title, Paragraph } = Typography;
const { TextArea } = Input;
const { Panel } = Collapse;
const { TabPane } = Tabs;

// 配置状态指示器组件
const ConfigStatusIndicator = ({ configValid, configWarnings, language }) => {
  const t = i18n[language];

  if (configValid) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <CheckCircleOutlined style={{ color: '#52c41a', fontSize: 16 }} />
        <Text type="success">{t.configValid || (language === 'en' ? 'Configuration Valid' : '配置有效')}</Text>
      </div>
    );
  } else if (configWarnings && configWarnings.length > 0) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <ExclamationCircleOutlined style={{ color: '#faad14', fontSize: 16 }} />
          <Text type="warning">{language === 'en' ? 'Configuration has warnings' : '配置存在警告'}</Text>
        </div>
        <div style={{ paddingLeft: 24, fontSize: 12 }}>
          {configWarnings.map((warning, index) => (
            <Text key={index} type="secondary">{warning}</Text>
          ))}
        </div>
      </div>
    );
  } else {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <ExclamationCircleOutlined style={{ color: '#f5222d', fontSize: 16 }} />
        <Text type="danger">{language === 'en' ? 'Configuration Invalid' : '配置无效'}</Text>
      </div>
    );
  }
};



// 优化卡片组件
const EnhancedCard = ({ title, icon, children, extra, bordered = true, className, ...props }) => (
  <Card
    className={`enhanced-card ${className || ''}`}
    title={
      title ? (
        <Space>
          {icon}
          <Text strong>{title}</Text>
        </Space>
      ) : null
    }
    bordered={bordered}
    size="small"
    extra={extra}
    {...props}
    style={{
      marginBottom: 16,
      borderRadius: 8,
      boxShadow: '0 2px 8px rgba(0, 0, 0, 0.06)',
      ...props.style
    }}
  >
    {children}
  </Card>
);



// 独立的 ScenarioSelector 组件
const ScenarioSelector = ({
  selectedScenario,
  onScenarioChange,
  language,
  scenarios = SCENARIO_TEMPLATES
}) => {
  const [searchText, setSearchText] = useState('');

  // 过滤场景
  const filteredScenarios = useMemo(() => {
    if (!searchText) return Object.entries(scenarios);

    return Object.entries(scenarios).filter(([key, scenario]) =>
      scenario.title.toLowerCase().includes(searchText.toLowerCase()) ||
      scenario.description.toLowerCase().includes(searchText.toLowerCase()) ||
      scenario.type.toLowerCase().includes(searchText.toLowerCase())
    );
  }, [scenarios, searchText]);

  // 场景类型颜色映射
  const typeColors = {
    portfolio: '#52c41a',
    engineering: '#1890ff',
    production: '#fa8c16',
    scheduling: '#722ed1',
    supply_chain: '#13c2c2',
    multiobjective: '#eb2f96',
    inverse: '#fa541c',
    benchmark: '#fadb14',
    constrained: '#a0d911',
    surrogate: '#2f54eb',
    comparison: '#722ed1',
    metaheuristic: '#faad14',
    other: '#d9d9d9'
  };

  // 生成详细的Tooltip内容
  const getTooltipContent = (scenario) => {
    const isChinese = language === 'zh';

    return (
      <div style={{ maxWidth: 400, fontSize: 12, lineHeight: 1.5 }}>
        <div style={{ fontWeight: 'bold', marginBottom: 8, color: '#1890ff' }}>
          {isChinese ? '问题详情' : 'Problem Details'}
        </div>

        {/* 目标描述 */}
        <div style={{ marginBottom: 6 }}>
          <span style={{ fontWeight: 'bold', color: '#52c41a' }}>
            {isChinese ? '优化目标：' : 'Optimization Goal: '}
          </span>
          {getGoalDescription(scenario, isChinese)}
        </div>

        {/* 参数说明 */}
        <div style={{ marginBottom: 6 }}>
          <span style={{ fontWeight: 'bold', color: '#fa8c16' }}>
            {isChinese ? '决策变量：' : 'Decision Variables: '}
          </span>
          {getVariablesDescription(scenario, isChinese)}
        </div>

        {/* 约束条件 */}
        {scenario.content?.constraints && scenario.content.constraints.length > 0 && (
          <div style={{ marginBottom: 6 }}>
            <span style={{ fontWeight: 'bold', color: '#eb2f96' }}>
              {isChinese ? '约束条件：' : 'Constraints: '}
            </span>
            {getConstraintsDescription(scenario, isChinese)}
          </div>
        )}

        {/* 优化算法 */}
        <div>
          <span style={{ fontWeight: 'bold', color: '#2f54eb' }}>
            {isChinese ? '优化算法：' : 'Optimization Algorithm: '}
          </span>
          {getAlgorithmDescription(scenario, isChinese)}
        </div>
      </div>
    );
  };

  // 获取目标描述
  const getGoalDescription = (scenario, isChinese) => {
    const { title, content } = scenario;

    switch (title) {
      case '投资组合优化（夏普比率最大化）':
        return isChinese ? '最大化夏普比率，平衡投资组合的收益率和风险' :
               'Maximize Sharpe ratio, balancing portfolio return and risk';
      case '工程设计优化（成本最小化）':
        return isChinese ? '最小化材料成本和加工成本，优化圆柱形容器设计' :
               'Minimize material and processing costs for cylindrical container design';
      case 'Rosenbrock函数优化（单目标）':
        return isChinese ? '最小化经典Rosenbrock测试函数值' :
               'Minimize classic Rosenbrock test function value';
      case '单位圆内最大化x+y（带约束）':
        return isChinese ? '在单位圆约束下最大化x+y的值' :
               'Maximize x+y within unit circle constraint';
      case '代理模型拟合精度测试':
        return isChinese ? '通过LightGBM模型拟合目标变量，最小化预测误差' :
               'Fit target variable with LightGBM model, minimize prediction error';
      case 'ZDT1多目标优化':
        return isChinese ? '同时最小化两个冲突的目标函数，寻找帕累托最优解' :
               'Minimize two conflicting objectives simultaneously, find Pareto optimal solutions';
      case '工厂排班优化':
        return isChinese ? '最小化人员排班总成本，满足生产需求' :
               'Minimize total staffing cost while meeting production requirements';
      case '生产调度优化':
        return isChinese ? '最小化生产成本和加班成本，优化生产批量和速率' :
               'Minimize production and overtime costs, optimize batch size and production rate';
      case '供应链生产分配优化':
        return isChinese ? '最小化多工厂生产总成本，满足总产量要求' :
               'Minimize total production cost across multiple factories, meet total output requirements';
      case '风险平价投资组合优化':
        return isChinese ? '使各资产贡献的风险相等，实现风险均衡分配' :
               'Equalize risk contribution from each asset, achieve balanced risk allocation';
      case '均值-方差投资组合优化':
        return isChinese ? '在给定收益率下最小化风险，或在给定风险下最大化收益率' :
               'Minimize risk for given return, or maximize return for given risk';
      case '目标波动率投资组合优化':
        return isChinese ? '在目标波动率约束下最大化投资组合收益率' :
               'Maximize portfolio return under target volatility constraint';
      case '优化算法性能对比':
        return isChinese ? '对比不同优化算法在Sphere函数上的优化性能' :
               'Compare optimization performance of different algorithms on Sphere function';
      case '历史生产数据排程优化':
        return isChinese ? '最大化生产效率评分，基于历史数据优化生产参数' :
               'Maximize production efficiency score, optimize parameters based on historical data';
      case '逆向优化-工艺参数（连续变量）':
        return isChinese ? '反推实现目标质量值的最优工艺参数组合' :
               'Inverse optimization to find optimal process parameters for target quality';
      case '逆向优化-供应链设计（离散变量）':
        return isChinese ? '寻找满足服务水平目标的最优供应链设计方案' :
               'Find optimal supply chain design to meet service level target';
      case '蚁群算法-物流路径规划':
        return isChinese ? '最小化物流配送总路径成本，优化访问顺序' :
               'Minimize total logistics route cost, optimize visiting sequence';
      case 'NSGA2-生产质量效率多目标优化':
        return isChinese ? '同时最大化产品质量和生产效率，最小化能源消耗' :
               'Maximize product quality and production efficiency, minimize energy consumption';
      case 'NSGA2-设备维护多目标优化':
        return isChinese ? '最小化维护成本和能源浪费，最大化设备可用性' :
               'Minimize maintenance cost and energy waste, maximize equipment availability';
      case 'NSGA2-供应链库存多目标优化':
        return isChinese ? '最小化库存成本和牛鞭效应，最大化服务水平' :
               'Minimize inventory cost and bullwhip effect, maximize service level';
      case 'NSGA2-质量工艺参数多目标优化':
        return isChinese ? '最小化缺陷率和能源消耗，最大化生产吞吐量' :
               'Minimize defect rate and energy consumption, maximize production throughput';
      case 'NSGA2-ZDT1简单多目标优化':
        return isChinese ? 'ZDT1测试问题的多目标优化，验证NSGA2算法性能' :
               'Multi-objective optimization of ZDT1 test problem, validate NSGA2 performance';
      default:
        return isChinese ? '优化目标详见具体场景' : 'Optimization goal details in specific scenario';
    }
  };

  // 获取变量描述
  const getVariablesDescription = (scenario, isChinese) => {
    const { title, content } = scenario;

    switch (title) {
      case '投资组合优化（夏普比率最大化）':
        return isChinese ? '三种资产的权重分配（w0, w1, w2）' :
               'Weight allocation for three assets (w0, w1, w2)';
      case '工程设计优化（成本最小化）':
        return isChinese ? '圆柱半径(x1)、长度(x2)、壁厚(x3)' :
               'Cylinder radius(x1), length(x2), wall thickness(x3)';
      case 'Rosenbrock函数优化（单目标）':
        return isChinese ? '二维连续变量(x1, x2)' : '2D continuous variables (x1, x2)';
      case '单位圆内最大化x+y（带约束）':
        return isChinese ? '二维坐标变量(x, y)' : '2D coordinate variables (x, y)';
      case '代理模型拟合精度测试':
        return isChinese ? '三个特征变量(feature1, feature2, feature3)' :
               'Three feature variables (feature1, feature2, feature3)';
      case 'ZDT1多目标优化':
        return isChinese ? '两个决策变量(x1, x2)' : 'Two decision variables (x1, x2)';
      case '工厂排班优化':
        return isChinese ? '工作日人员数量、周末人员数量' :
               'Weekday staff count, weekend staff count';
      case '生产调度优化':
        return isChinese ? '批量大小、生产速率、加班时长' :
               'Batch size, production rate, overtime hours';
      case '供应链生产分配优化':
        return isChinese ? '三个工厂的产量分配' : 'Production allocation for three factories';
      case '风险平价投资组合优化':
        return isChinese ? '四种资产的权重分配' : 'Weight allocation for four assets';
      case '均值-方差投资组合优化':
        return isChinese ? '股票、债券、REIT的权重分配' :
               'Weight allocation for stocks, bonds, REITs';
      case '目标波动率投资组合优化':
        return isChinese ? '成长型、价值型、防御型资产权重' :
               'Weights for growth, value, defensive assets';
      case '优化算法性能对比':
        return isChinese ? '三维连续变量(x1, x2, x3)' : '3D continuous variables (x1, x2, x3)';
      case '历史生产数据排程优化':
        return isChinese ? '日产能、安全库存、维护频率、工作班次' :
               'Daily capacity, safety stock, maintenance frequency, worker shifts';
      case '逆向优化-工艺参数（连续变量）':
        return isChinese ? '温度、压力、速度、冷却时间' :
               'Temperature, pressure, speed, cooling time';
      case '逆向优化-供应链设计（离散变量）':
        return isChinese ? '仓库规模选择、运输方式、库存策略' :
               'Warehouse size selection, transport mode, inventory strategy';
      case '蚁群算法-物流路径规划':
        return isChinese ? '8个地点的访问顺序' : 'Visiting sequence for 8 locations';
      case 'NSGA2-生产质量效率多目标优化':
        return isChinese ? '机器速度、温度、压力、操作员技能等级' :
               'Machine speed, temperature, pressure, operator skill level';
      case 'NSGA2-设备维护多目标优化':
        return isChinese ? '预防维护间隔、备件水平、维护团队规模、检查频率' :
               'Preventive interval, spare parts level, maintenance team size, inspection frequency';
      case 'NSGA2-供应链库存多目标优化':
        return isChinese ? '安全库存水平、再订货点、订货量、检查周期' :
               'Safety stock level, reorder point, order quantity, review period';
      case 'NSGA2-质量工艺参数多目标优化':
        return isChinese ? '工艺温度、压力、时间、材料粘度' :
               'Process temperature, pressure, time, material viscosity';
      case 'NSGA2-ZDT1简单多目标优化':
        return isChinese ? '两个决策变量(x1, x2)' : 'Two decision variables (x1, x2)';
      default:
        return isChinese ? `${content?.variables?.length || 0}个决策变量` :
               `${content?.variables?.length || 0} decision variables`;
    }
  };

  // 获取约束描述
  const getConstraintsDescription = (scenario, isChinese) => {
    const { title, content } = scenario;

    switch (title) {
      case '投资组合优化（夏普比率最大化）':
        return isChinese ? '权重总和等于1' : 'Sum of weights equals 1';
      case '工程设计优化（成本最小化）':
        return isChinese ? '体积约束、尺寸下限约束' : 'Volume constraints, minimum size constraints';
      case '单位圆内最大化x+y（带约束）':
        return isChinese ? 'x² + y² ≤ 1' : 'x² + y² ≤ 1';
      case '供应链生产分配优化':
        return isChinese ? '总产量约束、各工厂产能上限' :
               'Total production constraint, capacity limits for each factory';
      case '均值-方差投资组合优化':
        return isChinese ? '权重总和等于1、最低收益率约束' :
               'Sum of weights equals 1, minimum return constraint';
      case '目标波动率投资组合优化':
        return isChinese ? '权重总和等于1、波动率等于目标值' :
               'Sum of weights equals 1, volatility equals target';
      case '历史生产数据排程优化':
        return isChinese ? '产能满足需求、能源消耗上限' :
               'Capacity meets demand, energy consumption limit';
      case '逆向优化-工艺参数（连续变量）':
        return isChinese ? '温度压力总和约束' : 'Temperature-pressure sum constraint';
      case '蚁群算法-物流路径规划':
        return isChinese ? '每个地点访问且仅访问一次' : 'Each location visited exactly once';
      default:
        return isChinese ? `${content?.constraints?.length || 0}个约束条件` :
               `${content?.constraints?.length || 0} constraints`;
    }
  };

  // 获取算法描述
  const getAlgorithmDescription = (scenario, isChinese) => {
    const { content } = scenario;
    const optimizer = content?.optimizer || '未知算法';

    const algoMap = {
      'scipy': isChinese ? 'SciPy优化库（SLSQP）' : 'SciPy optimization (SLSQP)',
      'pso': isChinese ? '粒子群优化算法' : 'Particle Swarm Optimization',
      'nsga2': isChinese ? 'NSGA-II多目标遗传算法' : 'NSGA-II Multi-objective GA',
      'differential_evolution': isChinese ? '差分进化算法' : 'Differential Evolution',
      'ga': isChinese ? '遗传算法' : 'Genetic Algorithm',
      'ant_colony': isChinese ? '蚁群优化算法' : 'Ant Colony Optimization'
    };

    return algoMap[optimizer] || (isChinese ? `${optimizer}算法` : `${optimizer} algorithm`);
  };

  return (
    <EnhancedCard
      title={language === 'en' ? 'Scenario Templates' : '场景模板'}
      icon={<BookOutlined style={{ color: '#52c41a' }} />}
    >
      {/* 搜索框 */}
      <div style={{ marginBottom: 16 }}>
        <Input
          placeholder={language === 'en' ? 'Search scenarios...' : '搜索场景...'}
          prefix={<SearchOutlined />}
          value={searchText}
          onChange={(e) => setSearchText(e.target.value)}
          allowClear
        />
      </div>

      {/* 自定义配置卡片 */}
      <Card
        style={{
          marginBottom: 16,
          border: selectedScenario === 'custom' ? '2px solid #1890ff' : '1px solid #f0f0f0',
          cursor: 'pointer',
          borderRadius: 8
        }}
        onClick={() => onScenarioChange('custom')}
        hoverable
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <SettingOutlined style={{ fontSize: 24, color: '#1890ff' }} />
          <div style={{ flex: 1 }}>
            <Text strong style={{ fontSize: 16 }}>
              {language === 'en' ? 'Custom Configuration' : '自定义配置'}
            </Text>
            <div>
              <Text type="secondary">
                {language === 'en'
                  ? 'Build your own optimization configuration from scratch'
                  : '从头开始构建您自己的优化配置'
                }
              </Text>
            </div>
          </div>
          {selectedScenario === 'custom' && (
            <CheckCircleOutlined style={{ color: '#52c41a', fontSize: 20 }} />
          )}
        </div>
      </Card>

      {/* 场景模板卡片网格 */}
      <Row gutter={[16, 16]}>
        {filteredScenarios.map(([key, scenario]) => (
          <Col xs={24} sm={12} lg={8} key={key}>
            <Tooltip
              title={getTooltipContent(scenario)}
              placement="top"
              mouseEnterDelay={0.3}
              overlayStyle={{ borderRadius: 6 }}
            >
              <Card
                style={{
                  height: '100%',
                  border: selectedScenario === key ? '2px solid #1890ff' : '1px solid #f0f0f0',
                  cursor: 'pointer',
                  borderRadius: 8,
                  position: 'relative'
                }}
                onClick={() => onScenarioChange(key)}
                hoverable
                bodyStyle={{ padding: 16, height: '100%' }}
              >
                {/* 信息图标 */}
                <div style={{
                  position: 'absolute',
                  top: 8,
                  right: 8,
                  fontSize: 12,
                  color: '#1890ff',
                  opacity: 0.7
                }}>
                  <InfoCircleOutlined />
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                  {/* 标题和类型标签 */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
                    <Text strong style={{ fontSize: 14, flex: 1 }}>
                      {scenario.title}
                    </Text>
                    <Tag
                      color={typeColors[scenario.type] || typeColors.other}
                      style={{ margin: 0, fontSize: 10 }}
                    >
                      {scenario.type}
                    </Tag>
                  </div>

                  {/* 描述 */}
                  <div style={{ flex: 1, marginBottom: 12 }}>
                    <Text type="secondary" style={{ fontSize: 12, lineHeight: 1.4 }}>
                      {scenario.description}
                    </Text>
                  </div>

                  {/* 配置信息 */}
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 8 }}>
                    {scenario.content?.optimizer && (
                      <Tag color="blue" style={{ fontSize: 10, margin: 0 }}>
                        {scenario.content.optimizer}
                      </Tag>
                    )}
                    {scenario.content?.surrogateModel && scenario.content.surrogateModel !== 'noModel' && (
                      <Tag color="green" style={{ fontSize: 10, margin: 0 }}>
                        {scenario.content.surrogateModel}
                      </Tag>
                    )}
                    {scenario.content?.objectiveType && (
                      <Tag color="purple" style={{ fontSize: 10, margin: 0 }}>
                        {scenario.content.objectiveType === 'multi'
                          ? (language === 'en' ? 'Multi' : '多目标')
                          : (language === 'en' ? 'Single' : '单目标')
                        }
                      </Tag>
                    )}
                  </div>

                  {/* 底部统计信息 */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#999' }}>
                    <span>
                      {scenario.content?.variables?.length || 0} {language === 'en' ? 'vars' : '变量'}
                    </span>
                    <span>
                      {scenario.content?.objectives?.length || 0} {language === 'en' ? 'objs' : '目标'}
                    </span>
                    <span>
                      {scenario.content?.constraints?.length || 0} {language === 'en' ? 'constraints' : '约束'}
                    </span>
                  </div>

                  {/* 选中标识 */}
                  {selectedScenario === key && (
                    <div style={{ position: 'absolute', top: 8, right: 24 }}>
                      <CheckCircleOutlined style={{ color: '#52c41a' }} />
                    </div>
                  )}
                </div>
              </Card>
            </Tooltip>
          </Col>
        ))}
      </Row>

      {filteredScenarios.length === 0 && (
        <Empty
          description={language === 'en' ? 'No scenarios found' : '未找到相关场景'}
          style={{ padding: 40 }}
        />
      )}
    </EnhancedCard>
  );
};

// 变量配置项组件
const VariableConfigItem = ({ field, form, language, availableColumns, onDelete }) => {
  const t = i18n[language];
  const variableType = Form.useWatch(['variables', field.name, 'type'], form);

  return (
    <Row gutter={[8, 8]} key={field.key} style={{
      marginBottom: 8,
      padding: 12,
      background: '#f9f9f9',
      borderRadius: 6,
      border: '1px solid #f0f0f0'
    }}>
      <Col span={6}>
        <Form.Item
          {...field}
          name={[field.name, 'name']}
          label={t.variableName}
          rules={[{ required: true, message: language === 'en' ? 'Variable name is required' : '变量名称不能为空' }]}
          tooltip={{
            title: language === 'en' ? 'Name of the optimization variable' : '优化变量的名称',
            icon: <InfoCircleOutlined />
          }}
        >
          <Select
            placeholder={t.enterVariableName}
            size="small"
            showSearch
            getPopupContainer={trigger => trigger.parentNode}
            filterOption={(input, option) =>
              option.children.toLowerCase().indexOf(input.toLowerCase()) >= 0
            }
          >
            {availableColumns.map(col => (
              <Option key={col} value={col}>{col}</Option>
            ))}
          </Select>
        </Form.Item>
      </Col>
      <Col span={4}>
        <Form.Item
          {...field}
          name={[field.name, 'type']}
          label={t.variableType}
          rules={[{ required: true, message: language === 'en' ? 'Variable type is required' : '变量类型不能为空' }]}
          tooltip={{
            title: language === 'en' ? 'Type of the optimization variable' : '优化变量的类型',
            icon: <InfoCircleOutlined />
          }}
        >
          <Select size="small" getPopupContainer={trigger => trigger.parentNode}>
            <Option value="continuous">
              <Space><LineChartOutlined />{t.continuous}</Space>
            </Option>
            <Option value="categorical">
              <Space><SafetyOutlined />{t.categorical}</Space>
            </Option>
            <Option value="ordinal">
              <Space><SettingOutlined />{t.ordinal}</Space>
            </Option>
          </Select>
        </Form.Item>
      </Col>
      <Col span={9}>
        <Form.Item
          {...field}
          name={[field.name, 'bounds']}
          label={t.bounds}
          rules={[{ required: true, message: language === 'en' ? 'Bounds are required' : '取值范围不能为空' }]}
          tooltip={{
            title: language === 'en'
              ? variableType === 'continuous' ? 'Minimum and maximum values' : 'Possible values (comma separated)'
              : variableType === 'continuous' ? '最小值和最大值' : '可能的取值（逗号分隔）',
            icon: <InfoCircleOutlined />
          }}
        >
          {variableType === 'continuous' ? (
            <Input.Group compact style={{ width: '100%' }}>
              <Form.Item name={[field.name, 'bounds', 0]} noStyle>
                <InputNumber
                  placeholder={language === 'en' ? 'Min' : '最小值'}
                  style={{ width: '50%' }}
                  precision={2}
                  size="small"
                />
              </Form.Item>
              <Form.Item name={[field.name, 'bounds', 1]} noStyle>
                <InputNumber
                  placeholder={language === 'en' ? 'Max' : '最大值'}
                  style={{ width: '50%' }}
                  precision={2}
                  size="small"
                />
              </Form.Item>
            </Input.Group>
          ) : (
            <Select
              mode="tags"
              tokenSeparators={[',']}
              style={{ width: '100%' }}
              placeholder={language === 'en' ? 'Enter categories (comma separated)' : '输入分类值(逗号分隔)'}
              size="small"
              getPopupContainer={trigger => trigger.parentNode}
            />
          )}
        </Form.Item>
      </Col>
      <Col span={3} style={{ display: 'flex', alignItems: 'flex-end' }}>
        <Popconfirm
          title={language === 'en' ? 'Delete this variable?' : '确定删除这个变量吗？'}
          onConfirm={() => onDelete(field.name)}
          placement="topRight"
        >
          <Button danger size="small" icon={<DeleteOutlined />} block>
            {t.delete}
          </Button>
        </Popconfirm>
      </Col>
    </Row>
  );
};

// 数据预览组件
const DataPreview = ({ data, columns, fileName, language }) => {
  const [isCollapsed, setIsCollapsed] = useState(true);

  if (!data || data.length === 0 || !columns || columns.length === 0) {
    return (
      <Empty
        description={language === 'en' ? 'No data to display' : '暂无数据'}
        image={Empty.PRESENTED_IMAGE_SIMPLE}
      />
    );
  }

  const previewData = isCollapsed ? data.slice(0, 3) : data;

  return (
    <div style={{
      background: 'white',
      borderRadius: 6,
      padding: 16,
      border: '1px solid #f0f0f0',
      maxHeight: 400,
      overflow: 'auto'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <Space>
          <DatabaseOutlined style={{ color: '#1890ff' }} />
          <Text strong>{language === 'en' ? 'Data Preview' : '数据预览'}</Text>
          <Badge count={data.length} style={{ backgroundColor: '#1890ff' }} />
          <Badge count={`${columns.length} ${language === 'en' ? 'columns' : '列'}`} style={{ backgroundColor: '#52c41a' }} />
          {fileName && <Tag color="blue" icon={<FileTextOutlined />}>{fileName}</Tag>}
        </Space>
        <Tooltip title={isCollapsed ? (language === 'en' ? 'Expand all data' : '展开全部数据') : (language === 'en' ? 'Collapse data' : '收起数据')}>
          <Button
            type="text"
            size="small"
            onClick={() => setIsCollapsed(!isCollapsed)}
            icon={<ColumnHeightOutlined />}
          >
            {isCollapsed ? (language === 'en' ? 'Expand' : '展开') : (language === 'en' ? 'Collapse' : '收起')}
          </Button>
        </Tooltip>
      </div>

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
        scroll={{ x: true }}
        style={{ background: 'white' }}
      />
      {isCollapsed && data.length > 3 && (
        <div style={{ textAlign: 'center', marginTop: 8, padding: '8px' }}>
          <Text type="secondary">
            {language === 'en'
              ? `... ${data.length - 3} more rows not shown, click Expand to view all`
              : `... 还有 ${data.length - 3} 行数据未显示，点击展开查看全部`}
          </Text>
        </div>
      )}
    </div>
  );
};

// 配置摘要组件
const ConfigSummary = ({ config, language }) => {
  const t = i18n[language];

  return (
    <EnhancedCard
      title={language === 'en' ? 'Configuration Summary' : '配置摘要'}
      icon={<FileTextOutlined style={{ color: '#1890ff' }} />}
    >
      <Descriptions column={2} size="small" bordered>
        <Descriptions.Item label={t.optimizationMode}>
          {config.optimizationMode === 'training' ? t.trainingAndOptimization : t.inferenceOnly}
        </Descriptions.Item>
        <Descriptions.Item label={t.surrogateModel}>
          {t[config.surrogateModel] || config.surrogateModel}
        </Descriptions.Item>
        <Descriptions.Item label={t.algorithm}>
          {t[config.optimizer] || config.optimizer}
        </Descriptions.Item>
        <Descriptions.Item label={t.objectiveType}>
          {config.objectiveType === 'single' ? t.singleObjective : t.multiObjective}
        </Descriptions.Item>
        <Descriptions.Item label={language === 'en' ? 'Variables' : '变量数量'}>
          {config.variables ? config.variables.length : 0}
        </Descriptions.Item>
        <Descriptions.Item label={language === 'en' ? 'Objectives' : '目标数量'}>
          {config.objectives ? config.objectives.length : 0}
        </Descriptions.Item>
        <Descriptions.Item label={language === 'en' ? 'Constraints' : '约束数量'}>
          {config.constraints ? config.constraints.length : 0}
        </Descriptions.Item>
        <Descriptions.Item label={language === 'en' ? 'Scenario' : '场景'}>
          {config.scenario === 'custom' ? t.customConfig : config.scenario}
        </Descriptions.Item>
      </Descriptions>
    </EnhancedCard>
  );
};

// 独立的FileUploader组件，使用memo防止不必要的重渲染
const FileUploader = React.memo(({
  language,
  dataColumns,
  uploadedData,
  uploadedColumns,
  uploadedFileName,
  onFileUpload,
  parseUploadedFile
}) => {
  const [uploadLoading, setUploadLoading] = useState(false);
  const [urlModalVisible, setUrlModalVisible] = useState(false);
  const [urlLoading, setUrlLoading] = useState(false);
  const [urlInput, setUrlInput] = useState('');
  const [fileType, setFileType] = useState('csv');

  // 使用useMemo缓存计算结果
  const hasPreviousOperator = useMemo(() => dataColumns && dataColumns.length > 0, [dataColumns]);
  const hasUploadedData = useMemo(() => uploadedColumns && uploadedColumns.length > 0, [uploadedColumns]);

  // 使用useCallback缓存函数
  const handleFileUpload = useCallback(async (file) => {
    setUploadLoading(true);


    try {
      const parsedData = await parseUploadedFile(file);
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


      onFileUpload({ ...parsedData, filehash: tmpfilehash });
      message.success(`${language === 'en' ? 'File uploaded successfully' : '文件上传成功'}`);
    } catch (error) {
      message.error(`${language === 'en' ? 'File parsing failed' : '文件解析失败'}: ${error.message}`);
    } finally {
      setUploadLoading(false);
    }
    return false; // 阻止默认上传行为
  }, [language, parseUploadedFile, onFileUpload]);

  const handleUrlLoad = useCallback(async () => {
    if (!urlInput.trim()) {
      message.error(language === 'en' ? 'Please enter URL' : '请输入URL');
      return;
    }

    setUrlLoading(true);
    try {
      let tmpfilehash =null
      const response = await axios.post('/api/tablepipeline/load-from-url', {
        url: urlInput,
        file_type: fileType,
        preview_rows: 50, // 只获取前50行作为预览
        need_metadata: true
      }, {
        withCredentials: true,
        timeout: 60000 // 大数据可能需要更长时间
      });
      if (response.data.status === 'success') {
        const data = response.data.data;
        tmpfilehash = data.tables.table.file_hash;
      }


      if (response.ok) {
        const result = await response.json();
        if (result.status === 'success') {
          const data = result.data;
          onFileUpload({
            data: data.preview_data || [],
            columns: data.columns || [],
            fileName: urlInput.split('/').pop() || 'remote_data'
          });
          setUrlModalVisible(false);
          setUrlInput('');
          message.success(`${language === 'en' ? 'Successfully loaded data from URL' : '从URL成功加载数据'}`);
        }
      }
    } catch (error) {
      message.error(`${language === 'en' ? 'Failed to load data from URL' : '从URL加载数据失败'}`);
    } finally {
      setUrlLoading(false);
    }
  }, [urlInput, fileType, language, onFileUpload]);

  const availableColumns = useMemo(() => {
    return hasPreviousOperator ? dataColumns : (hasUploadedData ? uploadedColumns : []);
  }, [hasPreviousOperator, dataColumns, hasUploadedData, uploadedColumns]);

  return (
    <EnhancedCard
      title={language === 'en' ? 'Data Source' : '数据源'}
      icon={<UploadOutlined style={{ color: '#1890ff' }} />}
    >
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Space>
            <Upload
              accept=".csv,.json,.xlsx,.xls"
              showUploadList={false}
              beforeUpload={handleFileUpload}
              disabled={uploadLoading || hasPreviousOperator}
            >
              <Button
                icon={<UploadOutlined />}
                loading={uploadLoading}
                style={{ width: 180 }}
                size="large"
              >
                {hasPreviousOperator
                  ? (language === 'en' ? 'Using Previous Data' : '使用前序数据')
                  : (language === 'en' ? 'Upload File' : '上传文件')}
              </Button>
            </Upload>

            <Button
              icon={<GlobalOutlined />}
              loading={urlLoading}
              onClick={() => setUrlModalVisible(true)}
              disabled={hasPreviousOperator}
              style={{
                background: 'linear-gradient(135deg, #52c41a, #73d13d)',
                border: 'none',
                color: 'white',
                width: 180
              }}
              size="large"
            >
              {language === 'en' ? 'Load from URL' : '从URL加载'}
            </Button>
          </Space>
        </Col>

        {hasPreviousOperator && (
          <Col span={24}>
            <Alert
              message={language === 'en' ? 'Data Source' : '数据源'}
              description={`Using data from previous operator (${dataColumns.length} columns available)`}
              type="info"
              showIcon
            />
          </Col>
        )}

        {hasUploadedData && (
          <Col span={24}>
            <Alert
              message={language === 'en' ? 'Data Source Info' : '数据源信息'}
              description={`Successfully loaded ${uploadedData.length} records with ${uploadedColumns.length} columns`}
              type="success"
              showIcon
            />
          </Col>
        )}
      </Row>

      {(hasPreviousOperator || hasUploadedData) && (
        <div style={{ marginTop: 4 }}>
          <DataPreview
            data={hasPreviousOperator ? [] : uploadedData}
            columns={availableColumns}
            fileName={uploadedFileName}
            language={language}
          />
        </div>
      )}

      {/* 直接内联模态框，避免复杂状态管理 */}
      <Modal
        title={language === 'en' ? 'Load Data from URL' : '从URL加载数据'}
        open={urlModalVisible}
        onCancel={() => {
          setUrlModalVisible(false);
          setUrlInput('');
        }}
        onOk={handleUrlLoad}
        confirmLoading={urlLoading}
        destroyOnClose={true} // 关闭时销毁Modal，防止内存泄漏
      >
        <Form layout="vertical">
          <Form.Item label={language === 'en' ? 'Data URL' : '数据URL'}>
            <Input
              placeholder={language === 'en' ? 'Enter CSV/JSON file URL' : '输入CSV/JSON文件URL'}
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              onPressEnter={handleUrlLoad}
            />
          </Form.Item>
          <Form.Item label={language === 'en' ? 'File Type' : '文件类型'}>
            <Select value={fileType} onChange={setFileType}>
              <Option value="csv">CSV</Option>
              <Option value="json">JSON</Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>
    </EnhancedCard>
  );
});

const OptimizationConfig = ({
  form: externalForm,
  dataColumns = [],
  onRunOptimization
}) => {
  const [internalForm] = Form.useForm();
  const form = externalForm || internalForm;

  // 表单监听
  const optimizationMode = Form.useWatch('optimizationMode', form);
  const surrogateModel = Form.useWatch('surrogateModel', form);
  const optimizer = Form.useWatch('optimizer', form);
  const objectiveType = Form.useWatch('objectiveType', form);
  const targetVariable = Form.useWatch('targetVariable', form);
  const variables = Form.useWatch('variables', form) || [];
  const objectives = Form.useWatch('objectives', form) || [];
  const constraints = Form.useWatch('constraints', form) || [];
  const fixedVariables = Form.useWatch('fixedVariables', form) || [];
  const optimizerParams = Form.useWatch('optimizerParams', form) || {};
  const modelParams = Form.useWatch('modelParams', form) || {};
  const inverseTarget = Form.useWatch('inverseTarget', form);
  const inverseTolerance = Form.useWatch('inverseTolerance', form);
  const useExistingModel = Form.useWatch('useExistingModel', form);
  const modelPath = Form.useWatch('modelPath', form);

  // 状态管理
  const [language, setLanguage] = useState('zh');
  const [activeTab, setActiveTab] = useState('basic');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [selectedScenario, setSelectedScenario] = useState('custom');
  const [uploadedData, setUploadedData] = useState(null);
  const [uploadedColumns, setUploadedColumns] = useState([]);
  const [uploadedFileName, setUploadedFileName] = useState('');
  const [configStatus, setConfigStatus] = useState({ valid: false, warnings: [] });
  const [optimizationResult, setOptimizationResult] = useState(null);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [filehash, setFilehash] = useState(null);
  const [formattedExpression, setFormattedExpression] = useState('');

  const t = i18n[language];

  const handleFileUploadResult = useCallback((result) => {
    setUploadedData(result.data || []);
    setUploadedColumns(result.columns || []);
    setUploadedFileName(result.fileName || '');
    setFilehash( result.filehash || null );
  }, []);
  // 数据源判断
  const hasPreviousOperator = useMemo(() => dataColumns && dataColumns.length > 0, [dataColumns]);
  const hasUploadedData = useMemo(() => uploadedColumns && uploadedColumns.length > 0, [uploadedColumns]);
  const availableColumns = useMemo(() => {
    return hasPreviousOperator ? dataColumns : (hasUploadedData ? uploadedColumns : []);
  }, [hasPreviousOperator, dataColumns, hasUploadedData, uploadedColumns]);


  const parseUploadedFile = useCallback((file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onload = (e) => {
        try {
          const content = e.target.result;
          let parsedData = [];
          let columns = [];

          if (file.name.endsWith('.csv')) {
            const lines = content.split('\n').filter(line => line.trim());
            if (lines.length > 0) {
              columns = lines[0].split(',').map(col => col.trim());
              parsedData = lines.slice(1).map(line => {
                const values = line.split(',');
                const item = {};
                columns.forEach((col, index) => {
                  const value = values[index] ? values[index].trim() : '';
                  item[col] = isNaN(Number(value)) ? value : Number(value);
                });
                return item;
              }).filter(item => Object.values(item).some(val => val !== '' && val !== undefined));
            }
          } else if (file.name.endsWith('.json')) {
            parsedData = JSON.parse(content);
            if (Array.isArray(parsedData) && parsedData.length > 0) {
              columns = Object.keys(parsedData[0]);
            } else if (typeof parsedData === 'object') {
              columns = Object.keys(parsedData);
              parsedData = [parsedData];
            }
          }

          resolve({
            data: parsedData,
            columns: columns,
            fileName: file.name
          });

        } catch (error) {
          reject(error);
        }
      };

      reader.onerror = () => {
        reject(new Error('File reading failed'));
      };

      if (file.name.endsWith('.csv') || file.name.endsWith('.json')) {
        reader.readAsText(file);
      } else {
        reject(new Error('Unsupported file type'));
      }
    });
  }, []);

  // 处理文件上传结果

  // 配置预览
  const configPreview = useMemo(() => {
  // 直接使用当前的表单值
  return {
    optimizationMode: optimizationMode || 'direct',
    surrogateModel: surrogateModel || 'noModel',
    optimizer: optimizer,
    objectiveType: objectiveType || 'single',
    targetVariable: targetVariable || '',
    variables: variables || [],
    objectives: objectives || [],
    constraints: constraints || [],
    fixedVariables: fixedVariables || [],
    optimizerParams: optimizerParams || {},
    modelParams: modelParams || {},
    inverseOptimization: {
      target: inverseTarget,
      tolerance: inverseTolerance
    },
    modelPath: modelPath,
    scenario: selectedScenario,
    filehash: filehash,
    timestamp: new Date().toISOString(),
    version: '1.0.0'
  };
}, [
  optimizationMode, surrogateModel, optimizer, objectiveType, targetVariable,
  variables, objectives, constraints, fixedVariables, optimizerParams,
  modelParams, inverseTarget, inverseTolerance, modelPath, selectedScenario
]);

  // 初始化表单
  useEffect(() => {
  if (form) {
    const initialValues = {
      optimizationMode: 'training',
      objectiveType: 'single',
      surrogateModel: 'lightgbm',
      variables: [],
      objectives: [],
      constraints: [],
      fixedVariables: [],
      optimizerParams: {},
      modelParams: {}
    };

    form.setFieldsValue(initialValues);
  }
}, [form]);

  // 配置验证
 useEffect(() => {
  const validate = () => {
    const errors = [];
    const warnings = [];

    // 基本验证
    if (!variables || variables.length === 0) {
      errors.push(language === 'en' ? 'At least one variable is required' : '至少需要一个变量');
    }

    if (!objectives || objectives.length === 0) {
      errors.push(language === 'en' ? 'At least one objective is required' : '至少需要一个目标');
    }

    // 算法验证
    if (selectedScenario === 'custom' && !optimizer) {
      errors.push(language === 'en' ? 'Algorithm is required for custom scenarios' : '自定义场景需要选择算法');
    }

    // 训练模式验证
    if (optimizationMode === 'training') {
      if (surrogateModel !== 'noModel' && !targetVariable) {
        warnings.push(language === 'en' ? 'Target variable is recommended for training' : '训练模式建议设置目标变量');
      }

      if (surrogateModel === 'noModel' && (!uploadedData || uploadedData.length === 0)) {
        warnings.push(language === 'en'
          ? 'Direct optimization works better with historical data for expression evaluation'
          : '直接优化建议提供历史数据以便表达式评估'
        );
      }
    }

    // 推理模式验证
    if (optimizationMode === 'inference') {
      if (!modelPath) {
        errors.push(language === 'en' ? 'Model path is required for inference' : '推理模式下需要模型路径');
      }

      if (!uploadedData || uploadedData.length === 0) {
        warnings.push(language === 'en'
          ? 'Inference mode works better with some data for feature validation'
          : '推理模式建议提供数据用于特征验证'
        );
      }
    }

    // 多目标验证
    if (objectiveType === 'multi') {
      if (!optimizer || !['nsga2', 'moead'].includes(optimizer)) {
        warnings.push(language === 'en'
          ? 'NSGA-II or MOEA/D is recommended for multi-objective optimization'
          : '多目标优化建议使用NSGA-II或MOEA/D算法'
        );
      }

      if (objectives.length < 2) {
        warnings.push(language === 'en'
          ? 'Multi-objective optimization typically requires 2+ objectives'
          : '多目标优化通常需要2个或更多目标'
        );
      }
    }

    // 单目标多算法验证
    if (objectiveType === 'single' && objectives.length > 1) {
      warnings.push(language === 'en'
        ? 'Single objective mode with multiple objectives - only the first will be used'
        : '单目标模式下设置多个目标 - 仅第一个目标会被使用'
      );
    }

    setConfigStatus({
      valid: errors.length === 0,
      warnings: errors.length === 0 ? warnings : [],
      errors
    });
  };

  validate();
}, [variables, objectives, optimizationMode, surrogateModel, targetVariable,
    modelPath, objectiveType, optimizer, language, selectedScenario, uploadedData]);
  // 应用场景模板
  const applyScenarioTemplate = useCallback((scenarioKey) => {
      const template = SCENARIO_TEMPLATES[scenarioKey];
      if (template && template.content && form) {
        // 先清空现有字段
        form.setFieldsValue({
          variables: [],
          objectives: [],
          constraints: [],
          fixedVariables: [],
          optimizerParams: {},
          modelParams: {},
          inverseTarget: undefined,
          inverseTolerance: undefined
        });

        // 应用模板配置 - 从 content 字段获取配置
        setTimeout(() => {
          const templateContent = template.content;
          const templateData = {
            optimizationMode: templateContent.optimizationMode || 'direct',
            surrogateModel: templateContent.surrogateModel || 'noModel',
            optimizer: templateContent.optimizer,
            objectiveType: templateContent.objectiveType || 'single',
            targetVariable: templateContent.targetVariable || '',
            variables: templateContent.variables || [],
            objectives: templateContent.objectives || [],
            constraints: templateContent.constraints || [],
            fixedVariables: templateContent.fixedVariables || [],
            optimizerParams: templateContent.optimizerParams || {},
            modelParams: templateContent.modelParams || {},
            inverseTarget: templateContent.inverseTarget,
            inverseTolerance: templateContent.inverseTolerance
          };

          console.log('Applying template data:', templateData);
          form.setFieldsValue(templateData);

          message.success(
            `${language === 'en' ? 'Applied' : '已应用'} ${template.title} ${language === 'en' ? 'template' : '模板'}`
          );
        }, 100);
      } else {
        console.error('Template or template content not found:', scenarioKey, template);
        message.error(`${language === 'en' ? 'Failed to apply template' : '应用模板失败'}`);
      }
    }, [form, language]);

  // 场景变更处理
  const handleScenarioChange = useCallback((value) => {
    setSelectedScenario(value);
    if (value !== 'custom') {
      applyScenarioTemplate(value);
    }
  }, [applyScenarioTemplate]);

  // 导出配置
  const exportConfig = useCallback(() => {
    const config = {
      ...configPreview,
      timestamp: new Date().toISOString(),
      version: '1.0.0'
    };

    const dataStr = JSON.stringify(config, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });

    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `optimization_config_${new Date().getTime()}.json`;
    link.click();

    message.success(`${language === 'en' ? 'Configuration exported successfully' : '配置导出成功'}`);
  }, [configPreview, language]);

  // 渲染算法参数配置
  const renderAlgorithmParams = () => {
    if (!optimizer || !ALGORITHM_CONFIGS[optimizer]) return null;

    return (
      <EnhancedCard
        title={language === 'en' ? 'Algorithm Parameters' : '算法参数'}
        icon={<SettingOutlined style={{ color: '#722ed1' }} />}
      >
        <Row gutter={[16, 16]}>
          {ALGORITHM_CONFIGS[optimizer].params.map(param => (
            <Col xs={24} sm={12} md={8} key={param.name}>
              <Form.Item
                name={['optimizerParams', param.name]}
                label={param.label[language]}
                initialValue={param.default}
                tooltip={{
                  title: param.description?.[language] || '',
                  icon: <InfoCircleOutlined />
                }}
              >
                {param.type === 'number' ? (
                  <InputNumber
                    min={param.min}
                    max={param.max}
                    step={param.step || 1}
                    style={{ width: '100%' }}
                    size="small"
                  />
                ) : param.type === 'select' ? (
                  <Select
                    getPopupContainer={trigger => trigger.parentNode}
                    size="small"
                  >
                    {param.options.map(opt => (
                      <Option key={opt.value} value={opt.value}>
                        {opt.label[language]}
                      </Option>
                    ))}
                  </Select>
                ) : (
                  <Input size="small" />
                )}
              </Form.Item>
            </Col>
          ))}
        </Row>
      </EnhancedCard>
    );
  };

  // 渲染模型参数配置
  const renderModelParams = () => {
    if (!surrogateModel || surrogateModel === 'noModel' || !MODEL_CONFIGS[surrogateModel]) return null;

    return (
      <EnhancedCard
        title={language === 'en' ? 'Model Parameters' : '模型参数'}
        icon={<LineChartOutlined style={{ color: '#fa8c16' }} />}
      >
        <Row gutter={[16, 16]}>
          {MODEL_CONFIGS[surrogateModel].params.map(param => (
            <Col xs={24} sm={12} md={8} key={param.name}>
              <Form.Item
                name={['modelParams', param.name]}
                label={param.label[language]}
                initialValue={param.default}
                tooltip={{
                  title: param.description?.[language] || '',
                  icon: <InfoCircleOutlined />
                }}
              >
                {param.type === 'number' ? (
                  <InputNumber
                    min={param.min}
                    max={param.max}
                    step={param.step || 1}
                    style={{ width: '100%' }}
                    size="small"
                  />
                ) : param.type === 'select' ? (
                  <Select
                    getPopupContainer={trigger => trigger.parentNode}
                    size="small"
                  >
                    {param.options.map(opt => (
                      <Option key={opt.value} value={opt.value}>
                        {opt.label[language]}
                      </Option>
                    ))}
                  </Select>
                ) : (
                  <Input size="small" />
                )}
              </Form.Item>
            </Col>
          ))}
        </Row>
      </EnhancedCard>
    );
  };

  const generateMockResult = (config) => {
  const isMultiObjective = config.objectiveType === 'multi';

  // 生成最优变量
  const optimal_variables = {};
  config.variables.forEach(varItem => {
    if (varItem.type === 'continuous') {
      optimal_variables[varItem.name] =
        (varItem.bounds[0] + varItem.bounds[1]) / 2 +
        (Math.random() - 0.5) * (varItem.bounds[1] - varItem.bounds[0]) * 0.1;
    } else if (varItem.type === 'categorical' && varItem.categories) {
      optimal_variables[varItem.name] =
        varItem.categories[Math.floor(Math.random() * varItem.categories.length)];
    }
  });

  // 生成优化历史
  const optimization_history = Array(50).fill().map((_, i) => ({
    iteration: i + 1,
    best_fitness: 100 * Math.exp(-i / 10) + Math.random() * 10,
    current_fitness: 100 * Math.exp(-i / 15) + Math.random() * 20
  }));

  const ExpressionDisplay = ({ value, onChange, placeholder }) => {
  // 格式化显示：逗号换行，但保存原始数据
  const displayValue = value ? value.replace(/,/g, ',\n') : '';

  const handleChange = (e) => {
    // 移除显示用的换行，保存原始数据
    const rawValue = e.target.value.replace(/,\n/g, ',');
    onChange(rawValue);
  };

  return (
    <TextArea
      rows={6}
      value={displayValue}
      onChange={handleChange}
      placeholder={placeholder}
      style={{
        fontFamily: 'Monaco, Consolas, monospace',
        fontSize: '13px',
        lineHeight: '1.5',
        whiteSpace: 'pre-wrap'
      }}
    />
  );
};



  return {
    success: true,
    optimal_variables,
    optimal_objective: Math.random() * 100,
    optimization_history,
    pareto_front: isMultiObjective ?
      Array(20).fill().map(() => [
        Math.random() * 10,
        Math.random() * 10
      ]) : null,
    convergence: {
      converged: true,
      iterations: optimization_history.length,
      message: language === 'en' ? 'Converged successfully' : '成功收敛'
    },
    statistics: {
      function_evaluations: 1000 + Math.floor(Math.random() * 500),
      execution_time: 2.5 + Math.random() * 1.5,
      improvement_ratio: 0.85 + Math.random() * 0.1
    },
    message: language === 'en' ? 'Optimization completed successfully' : '优化成功完成'
  };
};

  // 执行优化
  const executeOptimization = useCallback(async (config) => {
  try {
    setIsOptimizing(true);

    // 显示加载状态
    message.loading({
      content: language === 'en' ? 'Running optimization...' : '正在运行优化...',
      key: 'optimization',
      duration: 0
    });

    // 准备请求数据
    const requestData = {
      config: config,
      data: uploadedData || null,
      timestamp: new Date().toISOString()
    };

    // 实际的后端API调用
    const response = await axios('/api/optimization/run', {

      body: JSON.stringify(requestData)
    },{withCredentials:true});

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();

    // 处理优化结果
    setOptimizationResult(result);

    message.success({
      content: language === 'en' ? 'Optimization completed!' : '优化完成！',
      key: 'optimization'
    });

    // 触发父组件回调
    if (onRunOptimization) {
      onRunOptimization(result);
    }

    return result;

  } catch (error) {
    console.error('Optimization error:', error);

    // 如果后端不可用，使用模拟结果
    console.warn('Backend unavailable, using mock result for demonstration');

    const mockResult = generateMockResult(config);
    setOptimizationResult(mockResult);

    message.warning({
      content: language === 'en'
        ? 'Using demo mode (backend unavailable)'
        : '使用演示模式（后端服务不可用）',
      key: 'optimization'
    });

    if (onRunOptimization) {
      onRunOptimization(mockResult);
    }

    return mockResult;
  } finally {
    setIsOptimizing(false);
  }
}, [language, uploadedData, onRunOptimization]);

  // 处理运行优化
  const handleRunOptimization = useCallback(async () => {
  if (!form || !configStatus.valid) {
    console.log('Form validation failed:', { form, configStatus });
    return;
  }

  try {
    const values = await form.validateFields();
    console.log('Form values:', values);

    // 构建完整的配置对象
    const optimizationConfig = {
      ...values,
      scenario: selectedScenario,
      uploadedFileName: uploadedFileName,
      dataColumns: availableColumns
    };

    console.log('Optimization config:', optimizationConfig);

    // 执行优化
    await executeOptimization(optimizationConfig);

  } catch (error) {
    console.error('Validation or execution error:', error);
    if (error.errorFields) {
      message.error(language === 'en'
        ? 'Please check the form for errors'
        : '请检查表单中的错误');
      console.log('Form validation errors:', error.errorFields);
    }
  }
}, [form, configStatus.valid, selectedScenario, uploadedFileName, availableColumns, executeOptimization, language]);

  const exportResult = useCallback((result) => {
      const dataStr = JSON.stringify(result, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });

      const link = document.createElement('a');
      link.href = URL.createObjectURL(dataBlob);
      link.download = `optimization_result_${new Date().getTime()}.json`;
      link.click();

      message.success(`${language === 'en' ? 'Results exported successfully' : '结果导出成功'}`);
    }, [language]);

  const OptimizationResultPanel = ({ result, language, onNewOptimization }) => {
  if (!result) return null;

  return (
    <div>
      <Result
        status="success"
        title={language === 'en' ? 'Optimization Completed' : '优化完成'}
        subTitle={result.message}
        extra={[
          <Button type="primary" key="new" onClick={onNewOptimization}>
            {language === 'en' ? 'Run New Optimization' : '运行新的优化'}
          </Button>,
          <Button key="export" onClick={() => exportResult(result)}>
            {language === 'en' ? 'Export Results' : '导出结果'}
          </Button>
        ]}
      />

      {/* 最优解 */}
      <EnhancedCard
        title={language === 'en' ? 'Optimal Solution' : '最优解'}
        icon={<CheckCircleOutlined style={{ color: '#52c41a' }} />}
      >
        <Descriptions column={2} size="small" bordered>
          {Object.entries(result.optimal_variables).map(([key, value]) => (
            <Descriptions.Item key={key} label={key}>
              {typeof value === 'number' ? value.toFixed(4) : String(value)}
            </Descriptions.Item>
          ))}
          <Descriptions.Item label={language === 'en' ? 'Objective Value' : '目标函数值'}>
            {result.optimal_objective?.toFixed(4)}
          </Descriptions.Item>
        </Descriptions>
      </EnhancedCard>

      {/* 收敛历史 */}
      {result.optimization_history && (
        <EnhancedCard
          title={language === 'en' ? 'Convergence History' : '收敛历史'}
          icon={<LineChartOutlined style={{ color: '#1890ff' }} />}
        >
          <div style={{ height: 300 }}>
            {/* 这里可以集成图表组件显示收敛曲线 */}
            <div style={{
              padding: 40,
              textAlign: 'center',
              background: '#f9f9f9',
              borderRadius: 6
            }}>
              <LineChartOutlined style={{ fontSize: 48, color: '#ddd' }} />
              <div style={{ marginTop: 16 }}>
                <Text type="secondary">
                  {language === 'en'
                    ? 'Convergence chart would be displayed here'
                    : '收敛图表将在此显示'
                  }
                </Text>
              </div>
            </div>
          </div>
        </EnhancedCard>
      )}

      {/* 统计信息 */}
      {result.statistics && (
        <EnhancedCard
          title={language === 'en' ? 'Optimization Statistics' : '优化统计'}
          icon={<BarChartOutlined style={{ color: '#fa8c16' }} />}
        >
          <Row gutter={[16, 16]}>
            <Col xs={12} sm={8}>
              <Statistic
                title={language === 'en' ? 'Function Evaluations' : '函数评估次数'}
                value={result.statistics.function_evaluations}
                suffix="times"
              />
            </Col>
            <Col xs={12} sm={8}>
              <Statistic
                title={language === 'en' ? 'Execution Time' : '执行时间'}
                value={result.statistics.execution_time}
                precision={2}
                suffix="s"
              />
            </Col>
            <Col xs={12} sm={8}>
              <Statistic
                title={language === 'en' ? 'Improvement Ratio' : '改进比例'}
                value={result.statistics.improvement_ratio}
                precision={3}
              />
            </Col>
          </Row>
        </EnhancedCard>
      )}
    </div>
  );
};

  // 文件上传处理
  const handleFileUpload = useCallback((file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onload = (e) => {
        try {
          const content = e.target.result;
          let parsedData = [];
          let columns = [];

          if (file.name.endsWith('.csv')) {
            const lines = content.split('\n').filter(line => line.trim());
            if (lines.length > 0) {
              columns = lines[0].split(',').map(col => col.trim());
              parsedData = lines.slice(1).map(line => {
                const values = line.split(',');
                const item = {};
                columns.forEach((col, index) => {
                  const value = values[index] ? values[index].trim() : '';
                  item[col] = isNaN(Number(value)) ? value : Number(value);
                });
                return item;
              }).filter(item => Object.values(item).some(val => val !== '' && val !== undefined));
            }
          } else if (file.name.endsWith('.json')) {
            parsedData = JSON.parse(content);
            if (Array.isArray(parsedData) && parsedData.length > 0) {
              columns = Object.keys(parsedData[0]);
            } else if (typeof parsedData === 'object') {
              columns = Object.keys(parsedData);
              parsedData = [parsedData];
            }
          }

          setUploadedData(parsedData);
          setUploadedColumns(columns);
          setUploadedFileName(file.name);

          setTimeout(() => {
            message.success(`${language === 'en' ? 'File parsed successfully' : '文件解析成功'}`);
            resolve(true);
          }, 100);

        } catch (error) {
          console.error('File parsing error:', error);
          message.error(`${language === 'en' ? 'File parsing failed' : '文件解析失败'}: ${error.message}`);
          reject(error);
        }
      };

      reader.onerror = () => {
        message.error(`${language === 'en' ? 'File reading failed' : '文件读取失败'}`);
        reject(new Error('File reading failed'));
      };

      if (file.name.endsWith('.csv') || file.name.endsWith('.json')) {
        reader.readAsText(file);
      } else {
        reject(new Error('Unsupported file type'));
      }
    });
  }, [language]);





  return (
    <div className="optimization-config" style={{ padding: '20px', background: '#f5f5f5', minHeight: '40vh' }}>
      {/* 科技感标题栏 */}
      <div style={{
        background: 'linear-gradient(135deg, #0078ff 0%, #00c6ff 100%)',
        borderRadius: 12,
        padding: '24px 32px',
        marginBottom: 2,
        boxShadow: '0 8px 24px rgba(0, 120, 255, 0.2)',
        position: 'relative',
        overflow: 'hidden'
      }}>
        {/* 装饰元素 */}
        <div style={{
          position: 'absolute',
          top: '-50px',
          right: '-50px',
          width: '200px',
          height: '200px',
          background: 'rgba(255, 255, 255, 0.1)',
          borderRadius: '50%',
          filter: 'blur(50px)'
        }}></div>
        <div style={{
          position: 'absolute',
          bottom: '-30px',
          left: '-30px',
          width: '150px',
          height: '120px',
          background: 'rgba(255, 255, 255, 0.08)',
          borderRadius: '50%',
          filter: 'blur(40px)'
        }}></div>

        {/* 网格背景 */}
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundImage: 'linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)',
          backgroundSize: '20px 20px',
          opacity: 0.3
        }}></div>
        {/* linear-gradient(135deg, #667eea 0%, #764ba2 100%)很神奇的 */}
        {/* 内容 */}
        <div style={{ position: 'relative', zIndex: 1,marginTop:'0 px',marginBottom:'0 px' }}>
          <Row gutter={[9]} align="middle" style={{ marginTop:'0 px',marginBottom:'0 px' }}>
           <Col>
               <Row align="middle" gutter={9} style={{marginBottom: -9}}>
                <Col >
                  <ThunderboltOutlined style={{ fontSize: 30, color: 'white' }} />
                </Col>
                <Col flex="1" >
                  <Title level={4} style={{ color: 'white', margin: 0, fontSize: '24px' }}>
                    {t.optimizationConfig}
                  </Title>
                  <div style={{ marginTop: '0 px' }}>
                    <Space size={4}>
                      <StarFilled style={{ color: '#ffd666', fontSize: '10px' }} />
                      <Text style={{
                        color: 'rgba(255,255,255,0.95)',
                        fontSize: '13px',
                        fontWeight: '500',
                        letterSpacing: '0.5px'
                      }}>
                        {language === 'en'
                          ? 'Formula Engine • Multi-Objective Solver • Financial & Industrial Optimization'
                          : '科学计算: 公式推导 • 多目标求解 • 金融、工业寻优引擎'}
                      </Text>
                      <StarFilled style={{ color: '#ffd666', fontSize: '10px' }} />
                    </Space>
                  </div>
                </Col>
              </Row>
            </Col>
            <Col flex="auto" style={{ textAlign: 'right' }}>
              <Space>
                <ConfigStatusIndicator
                  configValid={configStatus.valid}
                  configWarnings={configStatus.warnings}
                  language={language}
                />
                <Space style={{
                  background: 'rgba(255, 255, 255, 0.15)',
                  borderRadius: 6,
                  padding: '4px 8px'
                }}>
                  <GlobalOutlined style={{ color: 'white' }} />
                  <Switch
                    checked={language === 'zh'}
                    onChange={(checked) => setLanguage(checked ? 'zh' : 'en')}
                    checkedChildren="中文"
                    unCheckedChildren="EN"
                    size="small"
                    style={{
                      backgroundColor: language === 'zh' ? '#00c6ff' : 'transparent',
                      borderColor: '#00c6ff'
                    }}
                  />
                </Space>
              </Space>
            </Col>
          </Row>
        </div>
      </div>

      {/* 文件上传组件 */}
      <FileUploader
        language={language}
        dataColumns={dataColumns}
        uploadedData={uploadedData}
        uploadedColumns={uploadedColumns}
        uploadedFileName={uploadedFileName}
        onFileUpload={handleFileUploadResult}
        parseUploadedFile={parseUploadedFile}
      />

      <Form form={form} layout="vertical" autoComplete="off">
        {/* 主要内容区域 */}
        <Row gutter={[16, 16]}>
          {/* 左侧主面板 */}
          <Col xs={24} lg={16}>
            {/* 模式选择 */}
            <EnhancedCard
  title={language === 'en' ? 'Optimization Mode' : '优化模式'}
  icon={<SettingOutlined style={{ color: '#1890ff' }} />}
>
  <Row gutter={[16, 16]}>
    <Col xs={24} md={12}>
      <Form.Item
        name="optimizationMode"
        label={t.optimizationMode}
        initialValue="training"
        tooltip={{
          title: language === 'en'
            ? 'Training: Build surrogate model + optimize | Inference: Use existing model for optimization only'
            : '训练模式：构建代理模型+优化 | 推理模式：仅使用现有模型进行优化',
          icon: <InfoCircleOutlined />
        }}
      >
        <Radio.Group>
          <Radio value="training">
            <Space>
              <LineChartOutlined />
              <div>
                <div>{t.trainingAndOptimization}</div>
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  {language === 'en' ? 'Build model + Optimize' : '构建模型 + 优化'}
                </Text>
              </div>
            </Space>
          </Radio>
          <Radio value="inference">
            <Space>
              <EyeOutlined />
              <div>
                <div>{t.inferenceOnly}</div>
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  {language === 'en' ? 'Use trained model only' : '仅使用训练好的模型'}
                </Text>
              </div>
            </Space>
          </Radio>
        </Radio.Group>
      </Form.Item>
    </Col>
  </Row>

          {optimizationMode === 'inference' && (
            <Alert
              message={language === 'en' ? 'Inference Mode Configuration' : '推理模式配置'}
              description={
                language === 'en'
                  ? 'In this mode, you need to provide a pre-trained model. The optimization will use this model for predictions without retraining.'
                  : '在此模式下，您需要提供预训练模型。优化将使用此模型进行预测而无需重新训练。'
              }
              type="info"
              showIcon
              style={{ marginBottom: 16 }}
            />
          )}
        </EnhancedCard>

            {/* 场景选择 */}
            <ScenarioSelector
              selectedScenario={selectedScenario}
              onScenarioChange={handleScenarioChange}
              language={language}
              t={t}
            />

            <Tabs
              activeKey={activeTab}
              onChange={setActiveTab}
              getPopupContainer={trigger => trigger.parentNode}
              tabBarExtraContent={
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 20,
                  marginTop: 2,
                  padding: 4,
                  borderRadius: 12,
                  boxShadow: '0 4px 12px rgba(0, 120, 255, 0.1)',
                  width: '100%'
                }}>

                  <Button
                    type="primary"
                    size="large"
                    icon={<ExperimentOutlined />}
                    onClick={() => {
                      console.log('按钮点击触发', {
                        configStatus: configStatus,
                        isOptimizing: isOptimizing,
                        formValid: form?.isFieldsTouched(true)
                      });
                      handleRunOptimization();
                    }}
                    // 临时注释掉disabled进行调试
                    // disabled={!configStatus.valid || isOptimizing}
                    loading={isOptimizing}
                    style={{
                      padding: '0 20px',
                      height: 28,
                      fontSize: 16,
                      borderRadius: 8,
                      background: 'linear-gradient(135deg, #0078ff 0%, #00c6ff 100%)',
                      border: 'none',
                      boxShadow: '0 4px 12px rgba(0, 120, 255, 0.3)'
                    }}
                  >
                    <Space>
                      <ApiOutlined />
                      {t.runOptimization}
                    </Space>
                  </Button>

                  {/* 添加调试信息 */}
                  <div style={{ fontSize: 12, color: '#666' }}>
                    配置有效: {configStatus.valid ? '是' : '否'} |
                    正在优化: {isOptimizing ? '是' : '否'}
                  </div>
                  {/* 执行按钮 - 临时移除disabled调试 */}
                  {/* 高级设置按钮 */}
                  <Button
                    type="default"
                    icon={showAdvanced ? <ExclamationCircleOutlined /> : <SettingOutlined />}
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    style={{
                      padding: '0 20px',
                      height: 40,
                      borderRadius: 8,
                      border: '1px solid #e6f7ff',
                      background: 'rgba(255, 255, 255, 0.8)'
                    }}
                  >
                    {showAdvanced ? t.basicSettings : t.advancedSettings}
                  </Button>
                </div>
              }
            >
              {/* 基础配置 */}
              <TabPane tab={t.basicConfig} key="basic"   >
                {/* 模型配置 */}
                <EnhancedCard
                  title={t.modelSettings}
                  icon={<LineChartOutlined style={{ color: '#fa8c16' }} />}
                >
                  <Row gutter={[16, 16]}>
                    <Col xs={24} md={8}>
                      <Form.Item
                        name="surrogateModel"
                        label={t.surrogateModel}
                        initialValue="lightgbm"
                        rules={[{ required: true, message: language === 'en' ? 'Model type is required' : '模型类型不能为空' }]}
                      >
                        <Select getPopupContainer={trigger => trigger.parentNode}>
                          <Option value="noModel">
                            <Space><SafetyOutlined />{t.noModel}</Space>
                          </Option>
                          <Option value="lightgbm">
                            <Space><LineChartOutlined />{t.lightgbm}</Space>
                          </Option>
                          <Option value="gaussian">
                            <Space><LineChartOutlined />{t.gaussian}</Space>
                          </Option>
                          <Option value="randomForest">
                            <Space><LineChartOutlined />{t.randomForest}</Space>
                          </Option>
                          <Option value="xgboost">
                            <Space><LineChartOutlined />{t.xgboost}</Space>
                          </Option>
                          <Option value="mlp">
                            <Space><LineChartOutlined />{t.mlp}</Space>
                          </Option>
                        </Select>
                      </Form.Item>
                    </Col>
                    <Col xs={24} md={8}>
                      <Form.Item
                        name="targetVariable"
                        label={t.targetVariable}
                        rules={[{
                          required: optimizationMode === 'training' && surrogateModel !== 'noModel' && selectedScenario === 'custom',
                          message: language === 'en' ? 'Target variable is required' : '目标变量不能为空'
                        }]}
                      >
                        <Select
                          placeholder={t.selectTargetVariable}
                          allowClear
                          getPopupContainer={trigger => trigger.parentNode}
                        >
                          {availableColumns.map(col => (
                            <Option key={col} value={col}>{col}</Option>
                          ))}
                        </Select>
                      </Form.Item>
                    </Col>
                    <Col xs={24} md={8}>
                      <Form.Item
                        name="objectiveType"
                        label={t.objectiveType}
                        initialValue="single"
                      >
                        <Radio.Group>
                          <Radio value="single">
                            <Space><LineChartOutlined />{t.singleObjective}</Space>
                          </Radio>
                          <Radio value="multi">
                            <Space><LineChartOutlined />{t.multiObjective}</Space>
                          </Radio>
                        </Radio.Group>
                      </Form.Item>
                    </Col>
                  </Row>

                  {showAdvanced && renderModelParams()}
                </EnhancedCard>

                {/* 变量设置 */}
                <EnhancedCard
                  title={t.variableSettings}
                  icon={<SettingOutlined style={{ color: '#52c41a' }} />}
                >
                  <Form.List name="variables">
                    {(fields, { add, remove }) => (
                      <>
                        {fields.length > 0 ? (
                          fields.map((field, index) => (
                            <VariableConfigItem
                              key={field.key}
                              field={field}
                              form={form}
                              language={language}
                              availableColumns={availableColumns}
                              onDelete={remove}
                            />
                          ))
                        ) : (
                          <Empty
                            description={language === 'en' ? 'No variables defined yet' : '尚未定义变量'}
                            style={{ padding: 20 }}
                          />
                        )}
                        <Form.Item style={{ marginTop: 16 }}>
                          <Button
                            type="dashed"
                            onClick={() => add()}
                            block
                            icon={<PlusOutlined />}
                            disabled={availableColumns.length === 0}
                          >
                            {t.addVariable}
                          </Button>
                          {availableColumns.length === 0 && (
                                                        <Text type="secondary" style={{ fontSize: '12px', display: 'block', marginTop: 8 }}>
                              {language === 'en' ? 'Please upload data first to define variables' : '请先上传数据以定义变量'}
                            </Text>
                          )}
                        </Form.Item>
                      </>
                    )}
                  </Form.List>
                </EnhancedCard>

                <EnhancedCard
                  title={t.algorithmSettings}
                  icon={<ThunderboltOutlined style={{ color: '#722ed1' }} />}
                >
                  <Row gutter={[16, 16]}>
                    <Col xs={24} md={12}>
                      <Form.Item
                        name="optimizer"
                        label={t.algorithm}
                        rules={[{
                          required: selectedScenario === 'custom',
                          message: language === 'en' ? 'Algorithm is required' : '算法不能为空'
                        }]}
                      >
                        <Select getPopupContainer={trigger => trigger.parentNode} allowClear>
                          <OptGroup label={language === 'en' ? 'Single-objective' : '单目标算法'}>
                            <Option value="pso">{t.pso}</Option>
                            <Option value="ga">{t.ga}</Option>
                            <Option value="bayesian">{t.bayesian}</Option>
                            <Option value="scipy">{t.scipy}</Option>
                            <Option value="cvxpy">{t.cvxpy}</Option>
                            <Option value="differential_evolution">
                              {language === 'en' ? 'Differential Evolution' : '差分进化'}
                            </Option>
                          </OptGroup>
                          <OptGroup label={language === 'en' ? 'Multi-objective' : '多目标算法'}>
                            <Option value="nsga2">{t.nsga2}</Option>
                            <Option value="moead">{t.moead}</Option>
                          </OptGroup>
                        </Select>
                      </Form.Item>
                    </Col>
                    <Col xs={24} md={12}>
                      <Form.Item
                        name="optimizationMode"
                        label={language === 'en' ? 'Optimization Mode' : '优化模式'}
                        initialValue="direct"
                      >
                        <Select getPopupContainer={trigger => trigger.parentNode}>
                          <Option value="direct">
                            {language === 'en' ? 'Direct Optimization' : '直接优化'}
                          </Option>
                          <Option value="inverse">
                            {language === 'en' ? 'Inverse Optimization' : '逆向优化'}
                          </Option>
                        </Select>
                      </Form.Item>
                    </Col>
                  </Row>

                  {showAdvanced && optimizer && renderAlgorithmParams()}
                </EnhancedCard>


                <EnhancedCard
                  title={t.objectiveSettings}
                  icon={<BarChartOutlined style={{ color: '#722ed1' }} />}
                >



                 <Form.List name="objectives">
                  {(fields, { add, remove }) => (
                    <>
                      {fields.map((field, index) => (
                        <div key={field.key} style={{
                          marginBottom: 16,
                          padding: 12,
                          background: '#f9f9f9',
                          borderRadius: 6,
                          border: '1px solid #f0f0f0'
                        }}>
                          <Row gutter={[8, 8]}>
                            <Col xs={24} sm={8}>
                              <Form.Item
                                {...field}
                                name={[field.name, 'name']}
                                label={t.objectiveName}
                                rules={[{ required: true }]}
                              >
                                <Input placeholder="e.g., sharpe, cost" size="small" />
                              </Form.Item>
                            </Col>
                            <Col xs={24} sm={6}>
                              <Form.Item
                                {...field}
                                name={[field.name, 'type']}
                                label={t.objectiveType}
                                rules={[{ required: true }]}
                              >
                                <Select size="small">
                                  <Option value="minimize">{t.minimize}</Option>
                                  <Option value="maximize">{t.maximize}</Option>
                                </Select>
                              </Form.Item>
                            </Col>
                            <Col xs={24} sm={8}>
                              <Form.Item
                                {...field}
                                name={[field.name, 'expressionType']}
                                label={language === 'en' ? 'Expression Type' : '表达式类型'}
                                initialValue="direct"
                              >
                                <Select size="small">
                                  <Option value="direct">
                                    {language === 'en' ? 'Direct Formula' : '直接公式'}
                                  </Option>
                                  <Option value="process">
                                    {language === 'en' ? 'Step-by-step' : '分步构建'}
                                  </Option>
                                </Select>
                              </Form.Item>
                            </Col>
                            <Col xs={24} sm={2} style={{ display: 'flex', alignItems: 'flex-end' }}>
                              <Popconfirm onConfirm={() => remove(field.name)}>
                                <Button danger size="small" icon={<DeleteOutlined />} block />
                              </Popconfirm>
                            </Col>
                          </Row>


                          {/* 表达式输入区域 */}
                          <Form.Item
                              {...field}
                              name={[field.name, 'expression']}
                              label={language === 'en' ? 'expression' : '表达式及构建过程'}
                              tooltip={{
                                title: language === 'en'
                                  ? 'Enter calculation steps separated by semicolons. Use MEAN(column), STD(column), COV(col1,col2) functions'
                                  : '输入计算步骤，用分号分隔。可以使用 MEAN(列名), STD(列名), COV(列1,列2) 等函数'
                              }}
                            >
                              <TextArea
                                rows={6}
                                onChange={(e) => {
                                  const rawValue = e.target.value;
                                  // 保存原始值到表单
                                  form.setFieldValue(['objectives', field.name, 'expression'], rawValue);
                                  // 格式化显示：在分号后添加换行
                                  const formatted = rawValue.replace(/;/g, ';\n');
                                  setFormattedExpression(formatted);
                                }}
                                onBlur={(e) => {
                                  // 确保保存的是原始值（不带额外换行）
                                  const rawValue = e.target.value.replace(/;\n/g, ';');
                                  form.setFieldValue(['objectives', field.name, 'expression'], rawValue);
                                }}
                                placeholder={
                                  language === 'en'
                                    ? `Example for portfolio optimization:
                            mean_return = MEAN(return_col1)*w1 + MEAN(return_col2)*w2;
                            variance = COV(return_col1,return_col1)*w1*w1 + COV(return_col1,return_col2)*w1*w2 + COV(return_col2,return_col1)*w2*w1 + COV(return_col2,return_col2)*w2*w2;
                            sharpe_ratio = (mean_return - 0.0001) / SQRT(variance)`
                                    : `投资组合优化示例：
                            mean_return = MEAN(收益列1)*w1 + MEAN(收益列2)*w2;
                            variance = COV(收益列1,收益列1)*w1*w1 + COV(收益列1,收益列2)*w1*w2 + COV(收益列2,收益列1)*w2*w1 + COV(收益列2,收益列2)*w2*w2;
                            sharpe_ratio = (mean_return - 0.0001) / SQRT(variance)`
                                }
                                style={{
                                  fontFamily: 'Monaco, Consolas, monospace',
                                  fontSize: '13px',
                                  lineHeight: '1.5',
                                  whiteSpace: 'pre-wrap'
                                }}
                              />
                            </Form.Item>

                        </div>
                      ))}
                      <Form.Item>
                        <Button type="dashed" onClick={() => add()} icon={<PlusOutlined />}>
                          {t.addObjective}
                        </Button>
                      </Form.Item>
                    </>
                  )}
                </Form.List>

                  {showAdvanced && optimizer && renderAlgorithmParams()}
                </EnhancedCard>


              </TabPane>

              {/* 高级配置 */}
              <TabPane tab={t.advancedConfig} key="advanced">
                {/* 约束设置 */}
                <EnhancedCard
                  title={t.constraintSettings}
                  icon={<SafetyOutlined style={{ color: '#faad14' }} />}
                >
                  <Alert
                    message={language === 'en' ? 'Constraint Expression Guide' : '约束表达式指南'}
                    description={
                      language === 'en'
                        ? 'Use variable names in expressions. Supported operators: +, -, *, /, **, ==, !=, <, <=, >, >='
                        : '在表达式中使用变量名。支持的运算符: +, -, *, /, **, ==, !=, <, <=, >, >='
                    }
                    type="info"
                    showIcon
                    style={{ marginBottom: 16, borderRadius: 6 }}
                  />

                  <Form.List name="constraints">
                    {(fields, { add, remove }) => (
                      <>
                        {fields.map((field, index) => (
                          <Row gutter={[8, 8]} key={field.key} style={{ marginBottom: 8 }}>
                            <Col xs={18} sm={16}>
                              <Form.Item
                                {...field}
                                name={[field.name, 'expression']}
                                label={language === 'en' ? 'Constraint Expression' : '约束表达式'}
                                rules={[{ required: true, message: language === 'en' ? 'Expression is required' : '表达式不能为空' }]}
                              >
                                <Input
                                  placeholder={language === 'en' ? 'e.g., x1 + x2 <= 10' : '例如: x1 + x2 <= 10'}
                                  size="small"
                                />
                              </Form.Item>
                            </Col>
                            <Col xs={4} sm={4}>
                              <Form.Item
                                {...field}
                                name={[field.name, 'type']}
                                label={language === 'en' ? 'Type' : '类型'}
                                initialValue="ineq"
                              >
                                <Select size="small" getPopupContainer={trigger => trigger.parentNode}>
                                  <Option value="ineq">
                                    <Space><SafetyOutlined />{language === 'en' ? 'Inequality' : '不等式'}</Space>
                                  </Option>
                                  <Option value="eq">
                                    <Space><SafetyOutlined />{language === 'en' ? 'Equality' : '等式'}</Space>
                                  </Option>
                                </Select>
                              </Form.Item>
                            </Col>
                            <Col xs={2} sm={2} style={{ display: 'flex', alignItems: 'flex-end' }}>
                              <Popconfirm
                                title={language === 'en' ? 'Delete this constraint?' : '确定删除这个约束吗？'}
                                onConfirm={() => remove(field.name)}
                                placement="topRight"
                              >
                                <Button danger size="small" icon={<DeleteOutlined />} block />
                              </Popconfirm>
                            </Col>
                          </Row>
                        ))}
                        <Form.Item>
                          <Button type="dashed" onClick={() => add()} icon={<PlusOutlined />}>
                            {t.addConstraint}
                          </Button>
                        </Form.Item>
                      </>
                    )}
                  </Form.List>
                </EnhancedCard>

                {/* 逆向优化 */}
                <EnhancedCard
                  title={t.inverseOptimization}
                  icon={<SettingOutlined style={{ color: '#722ed1' }} />}
                >
                  <Alert
                    message={t.inverseOptimization}
                    description={t.inverseScenario}
                    type="info"
                    showIcon
                    style={{ marginBottom: 16, borderRadius: 6 }}
                  />

                  <Row gutter={[16, 16]}>
                    <Col xs={24} sm={12}>
                      <Form.Item
                        name="inverseTarget"
                        label={language === 'en' ? 'Target Value' : '目标值'}
                      >
                        <InputNumber style={{ width: '100%' }} size="large" />
                      </Form.Item>
                    </Col>
                    <Col xs={24} sm={12}>
                      <Form.Item
                        name="inverseTolerance"
                        label={language === 'en' ? 'Tolerance' : '容差'}
                        initialValue={0.01}
                        tooltip={{
                          title: language === 'en' ? 'Allowed deviation from target' : '允许的目标值偏差',
                          icon: <InfoCircleOutlined />
                        }}
                      >
                        <InputNumber min={0} max={1} step={0.001} style={{ width: '100%' }} size="large" />
                      </Form.Item>
                    </Col>
                  </Row>
                  <Form.Item
                    name="inverseObjective"
                    label={language === 'en' ? 'Objective for Inverse Optimization' : '逆向优化目标'}
                    tooltip={{
                      title: language === 'en'
                        ? 'The objective function to use for inverse optimization'
                        : '用于逆向优化的目标函数'
                    }}
                  >
                    <Select allowClear>
                      {(objectives || []).map((obj, index) => (
                        <Option key={index} value={obj.name}>{obj.name}</Option>
                      ))}
                    </Select>
                  </Form.Item>

                  {/* 固定变量 */}
                  <EnhancedCard
                    title={language === 'en' ? 'Fixed Variables' : '固定变量'}
                    icon={<LockOutlined style={{ color: '#fa8c16' }} />}
                    bordered={false}
                  >
                    <Form.List name="fixedVariables">
                      {(fields, { add, remove }) => (
                        <>
                          {fields.map((field, index) => (
                            <Row gutter={[8, 8]} key={field.key} style={{ marginBottom: 8 }}>
                              <Col xs={12} sm={10}>
                                <Form.Item
                                  {...field}
                                  name={[field.name, 'name']}
                                  label={language === 'en' ? 'Variable Name' : '变量名'}
                                  rules={[{ required: true, message: language === 'en' ? 'Variable name is required' : '变量名不能为空' }]}
                                >
                                  <Select
                                    placeholder={language === 'en' ? 'Select variable' : '选择变量'}
                                    size="small"
                                    getPopupContainer={trigger => trigger.parentNode}
                                  >
                                    {variables.map((varItem, i) => (
                                      <Option key={i} value={varItem.name}>{varItem.name}</Option>
                                    ))}
                                  </Select>
                                </Form.Item>
                              </Col>
                              <Col xs={8} sm={8}>
                                <Form.Item
                                  {...field}
                                  name={[field.name, 'value']}
                                  label={language === 'en' ? 'Value' : '值'}
                                  rules={[{ required: true, message: language === 'en' ? 'Value is required' : '值不能为空' }]}
                                >
                                  <Input
                                    placeholder={language === 'en' ? 'Enter fixed value' : '输入固定值'}
                                    size="small"
                                  />
                                </Form.Item>
                              </Col>
                              <Col xs={2} sm={2} style={{ display: 'flex', alignItems: 'flex-end' }}>
                                <Popconfirm
                                  title={language === 'en' ? 'Delete this fixed variable?' : '确定删除这个固定变量吗？'}
                                  onConfirm={() => remove(field.name)}
                                  placement="topRight"
                                >
                                  <Button danger size="small" icon={<DeleteOutlined />} block />
                                </Popconfirm>
                              </Col>
                            </Row>
                          ))}
                          <Form.Item>
                            <Button
                              type="dashed"
                              onClick={() => add()}
                              icon={<PlusOutlined />}
                              disabled={!variables || variables.length === 0}
                            >
                              {language === 'en' ? 'Add Fixed Variable' : '添加固定变量'}
                            </Button>
                          </Form.Item>
                        </>
                      )}
                    </Form.List>
                  </EnhancedCard>
                </EnhancedCard>
              </TabPane>

              {/* 配置预览 */}
              <TabPane tab={t.preview} key="preview">
                <ConfigSummary config={configPreview} language={language} />

                <EnhancedCard
                  title={language === 'en' ? 'Configuration JSON' : '配置JSON'}
                  icon={<CodeOutlined style={{ color: '#722ed1' }} />}
                  extra={
                    <Button
                      icon={<ExportOutlined />}
                      type="primary"
                      size="small"
                      onClick={exportConfig}
                    >
                      {language === 'en' ? 'Export JSON' : '导出JSON'}
                    </Button>
                  }
                >
                  <Alert
                    message={language === 'en' ? 'Python Integration Guide' : 'Python集成指南'}
                    description={
                      language === 'en'
                        ? 'Export this configuration and use it with our Python optimization library for testing and deployment.'
                        : '导出此配置并与我们的Python优化库一起使用，用于测试和部署。'
                    }
                    type="info"
                    showIcon
                    style={{ marginBottom: 16, borderRadius: 6 }}
                  />

                  <div style={{
                    background: '#f9f9f9',
                    borderRadius: 6,
                    padding: 16,
                    maxHeight: 500,
                    overflow: 'auto'
                  }}>
                    <ReactJson
                      src={configPreview}
                      theme="rjv-default"
                      collapsed={1}
                      displayDataTypes={false}
                      style={{ fontSize: '12px' }}
                      enableClipboard={true}
                      onEdit={false}
                      onAdd={false}
                      onDelete={false}
                    />
                  </div>
                </EnhancedCard>
              </TabPane>

              {/* 优化结果 */}
              <TabPane tab={language === 'en' ? 'Optimization Result' : '优化结果'} key="result">
              {optimizationResult ? (
                <div>
                  <Result
                    status="success"
                    title={language === 'en' ? 'Optimization Completed' : '优化完成'}
                    subTitle={optimizationResult.message}
                    extra={[
                      <Button type="primary" key="new" onClick={() => setOptimizationResult(null)}>
                        {language === 'en' ? 'Run New Optimization' : '运行新的优化'}
                      </Button>,
                      <Button key="export" onClick={() => exportResult(optimizationResult)}>
                        {language === 'en' ? 'Export Results' : '导出结果'}
                      </Button>
                    ]}
                  />

                  {/* 最优解 */}
                  <EnhancedCard
                    title={language === 'en' ? 'Optimal Solution' : '最优解'}
                    icon={<CheckCircleOutlined style={{ color: '#52c41a' }} />}
                  >
                    <Descriptions column={2} size="small" bordered>
                      {Object.entries(optimizationResult.optimal_variables).map(([key, value]) => (
                        <Descriptions.Item key={key} label={key}>
                          {typeof value === 'number' ? value.toFixed(4) : String(value)}
                        </Descriptions.Item>
                      ))}
                      <Descriptions.Item label={language === 'en' ? 'Objective Value' : '目标函数值'}>
                        {optimizationResult.optimal_objective?.toFixed(4)}
                      </Descriptions.Item>
                    </Descriptions>
                  </EnhancedCard>

                  {/* 收敛历史 */}
                  {optimizationResult.optimization_history && (
                    <EnhancedCard
                      title={language === 'en' ? 'Convergence History' : '收敛历史'}
                      icon={<LineChartOutlined style={{ color: '#1890ff' }} />}
                    >
                      <div style={{ height: 300 }}>
                        <div style={{
                          padding: 40,
                          textAlign: 'center',
                          background: '#f9f9f9',
                          borderRadius: 6
                        }}>
                          <LineChartOutlined style={{ fontSize: 48, color: '#ddd' }} />
                          <div style={{ marginTop: 16 }}>
                            <Text type="secondary">
                              {language === 'en'
                                ? 'Convergence chart would be displayed here'
                                : '收敛图表将在此显示'
                              }
                            </Text>
                          </div>
                        </div>
                      </div>
                    </EnhancedCard>
                  )}

                  {/* 统计信息 */}
                  {optimizationResult.statistics && (
                    <EnhancedCard
                      title={language === 'en' ? 'Optimization Statistics' : '优化统计'}
                      icon={<BarChartOutlined style={{ color: '#fa8c16' }} />}
                    >
                      <Row gutter={[16, 16]}>
                        <Col xs={12} sm={8}>
                          <Statistic
                            title={language === 'en' ? 'Function Evaluations' : '函数评估次数'}
                            value={optimizationResult.statistics.function_evaluations}
                            suffix={language === 'en' ? 'times' : '次'}
                          />
                        </Col>
                        <Col xs={12} sm={8}>
                          <Statistic
                            title={language === 'en' ? 'Execution Time' : '执行时间'}
                            value={optimizationResult.statistics.execution_time}
                            precision={2}
                            suffix="s"
                          />
                        </Col>
                        <Col xs={12} sm={8}>
                          <Statistic
                            title={language === 'en' ? 'Improvement Ratio' : '改进比例'}
                            value={optimizationResult.statistics.improvement_ratio}
                            precision={3}
                          />
                        </Col>
                      </Row>
                    </EnhancedCard>
                  )}
                </div>
              ) : (
                <Empty
                  description={language === 'en' ? 'No optimization results yet' : '暂无优化结果'}
                  style={{ padding: 40 }}
                />
              )}
            </TabPane>
            </Tabs>
          </Col>

          {/* 右侧边栏 */}
          <Col xs={24} lg={8}>
            <div style={{ position: 'sticky', top: 20 }}>
              <ConfigSummary config={configPreview} language={language} />

              <EnhancedCard
                title={language === 'en' ? 'Optimization Tips' : '优化提示'}
                icon={<InfoCircleOutlined style={{ color: '#1890ff' }} />}
              >
                <div style={{ fontSize: 14, lineHeight: 1.6 }}>
                  <p>• {language === 'en'
                    ? 'For multi-objective problems, use NSGA-II or MOEA/D algorithm'
                    : '多目标问题建议使用NSGA-II或MOEA/D算法'}</p>
                  <p>• {language === 'en'
                    ? 'Gaussian Process is good for small datasets with smooth functions'
                    : '高斯过程适用于小数据集和光滑函数'}</p>
                  <p>• {language === 'en'
                    ? 'LightGBM/XGBoost works well for high-dimensional data'
                    : 'LightGBM/XGBoost适用于高维数据'}</p>
                  <p>• {language === 'en'
                    ? 'Bayesian Optimization is efficient for expensive function evaluations'
                    : '贝叶斯优化适用于昂贵的函数评估'}</p>
                </div>
              </EnhancedCard>

              <EnhancedCard
                  title={language === 'en' ? 'Algorithm Comparison' : '算法对比'}
                  icon={<LineChartOutlined style={{ color: '#fa8c16' }} />}
                >
                  <div style={{
                    background: '#f9f9f9',
                    borderRadius: 8,
                    padding: 16
                  }}>
                    {[
                      { name: 'CVXPY', category: '凸优化建模工具', finance: language === 'en' ? 'Portfolio optimization, Risk modeling' : '投资组合优化、风险建模', industrial: language === 'en' ? 'Control system design, Resource allocation' : '控制系统设计、资源分配', feature: language === 'en' ? 'Convex optimization, Easy modeling' : '凸优化、建模便捷', use_case: language === 'en' ? 'Convex optimization problems' : '凸优化问题' },
                       { name: 'Scipy', category: '经典数值优化', finance: language === 'en' ? 'Option pricing' : '期权定价', industrial: language === 'en' ? 'Parameter estimation' : '参数估计', feature: language === 'en' ? 'High precision' : '高精度', use_case: language === 'en' ? 'Smooth functions' : '光滑函数' },
                      { name: 'PSO', category: '粒子群优化', finance: language === 'en' ? 'Portfolio optimization' : '投资组合优化', industrial: language === 'en' ? 'Production scheduling' : '生产调度', feature: language === 'en' ? 'Fast convergence' : '收敛快', use_case: language === 'en' ? 'Complex systems' : '复杂系统' },
                      { name: 'GA', category: '遗传算法', finance: language === 'en' ? 'Risk management' : '风险管理', industrial: language === 'en' ? 'Process optimization' : '工艺优化', feature: language === 'en' ? 'Combinatorial problems' : '组合问题', use_case: language === 'en' ? 'Combinatorial optimization' : '组合优化' },
                      { name: 'NSGA-II', category: '非支配排序遗传算法', finance: language === 'en' ? 'Multi-objective portfolio' : '多目标投资组合', industrial: language === 'en' ? 'Manufacturing process' : '制造工艺', feature: language === 'en' ? 'Multiple solutions' : '多个解决方案', use_case: language === 'en' ? 'Conflicting objectives' : '冲突目标' },
                      { name: 'Bayesian', category: '贝叶斯优化', finance: language === 'en' ? 'Trading strategy' : '交易策略', industrial: language === 'en' ? 'Experiment design' : '实验设计', feature: language === 'en' ? 'Limited data' : '数据有限', use_case: language === 'en' ? 'Expensive functions' : '昂贵函数' },
                      { name: 'MOEA/D', category: '多目标进化算法', finance: language === 'en' ? 'Asset allocation' : '资产配置', industrial: language === 'en' ? 'Supply chain design' : '供应链设计', feature: language === 'en' ? 'Many objectives' : '多目标', use_case: language === 'en' ? '3+ objectives' : '3个以上目标' }

                    ].map((item, index) => (
                      <div key={index} style={{
                        background: 'white',
                        borderRadius: 6,
                        padding: 16,
                        marginBottom: 8,
                        boxShadow: '0 1px 3px rgba(0,0,0,0.05)'
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                          <h4 style={{ margin: 0, color: '#1890ff' }}>{item.name} - {item.category}</h4>
                          <span style={{
                            fontSize: 12,
                            background: '#e8f4f8',
                            padding: '2px 8px',
                            borderRadius: 4
                          }}>{item.use_case}</span>
                        </div>

                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
                          <div>
                            <strong style={{ fontSize: 12, color: '#666' }}>{language === 'en' ? 'Finance' : '金融'}</strong>
                            <p style={{ margin: '4px 0 0 0' }}>{item.finance}</p>
                          </div>
                          <div>
                            <strong style={{ fontSize: 12, color: '#666' }}>{language === 'en' ? 'Industrial' : '工业'}</strong>
                            <p style={{ margin: '4px 0 0 0' }}>{item.industrial}</p>
                          </div>
                          <div>
                            <strong style={{ fontSize: 12, color: '#666' }}>{language === 'en' ? 'Key Features' : '主要特点'}</strong>
                            <p style={{ margin: '4px 0 0 0' }}>{item.feature}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </EnhancedCard>
            </div>
          </Col>
        </Row>
      </Form>

      {/* 优化结果弹窗（如果需要） */}
      {optimizationResult && (
        <div style={{
          position: 'fixed',
          bottom: 20,
          right: 20,
          zIndex: 1000,
          background: 'white',
          borderRadius: 8,
          boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
          padding: 16,
          width: 300,
        }}>
          <Space>
            <CheckCircleOutlined style={{ color: '#52c41a', fontSize: 20 }} />
            <div>
              <Text strong>{language === 'en' ? 'Optimization Complete' : '优化完成'}</Text>
              <br />
              <Text type="secondary" style={{ fontSize: 12 }}>
                {language === 'en' ? 'Click to view detailed results' : '点击查看详细结果'}
              </Text>
            </div>
          </Space>
        </div>
      )}
    </div>
  );
};

export default OptimizationConfig;