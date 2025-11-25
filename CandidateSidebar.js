import React, { useState,useRef } from 'react';
import {
  Card, Tooltip, Button, Space, Tag, Input, Select
} from 'antd';
import {
  SearchOutlined, StarOutlined,
  DownOutlined, UpOutlined, AppstoreOutlined,
  DatabaseOutlined, RobotOutlined, LineChartOutlined,
  CodeOutlined, PictureOutlined, MessageOutlined,
  DragOutlined, DoubleRightOutlined
} from '@ant-design/icons';
import { candidateNodes, chineseLabels } from './Nodelist';

const { Option } = Select;

function CandidateSidebar({ onDragStart, onDoubleClick }) {
  const [openGroups, setOpenGroups] = useState(candidateNodes.map(group => group.group));
  const [displayMode, setDisplayMode] = useState('icon');
  const [searchText, setSearchText] = useState('');
  const [profession, setProfession] = useState('all');
  const [favorites, setFavorites] = useState(new Set());
  const [language, setLanguage] = useState('ch'); // 添加语言状态
  const [expandedItemGroups, setExpandedItemGroups] = useState({}); // 控制组内项目展开/折叠
  const lastClickTimeRef = useRef(0);

  // 国际化文本
  const i18n = {
    title: {
      ch: '智能算子库',
      en: 'AI Operators Library'
    },
    subtitle: {
      ch: '拖拽或双击添加算子',
      en: 'Drag or double-click to add operators'
    },
    search: {
      placeholder: {
        ch: '搜索算子...',
        en: 'Search operators...'
      }
    },
    displayMode: {
      icon: { ch: '图标', en: 'Icons' },
      list: { ch: '列表', en: 'List' },
      chinese: { ch: '中文', en: 'Chinese' }
    },
    profession: {
      all: { ch: '全部算子', en: 'All Operators' },
      dataAnalysis: { ch: '数据分析', en: 'Data Analysis' },
      machineLearning: { ch: '机器学习', en: 'Machine Learning' },
      largeModel: { ch: '大模型', en: 'Large Models' },
      agent: { ch: '智能体', en: 'AI Agents' },
      computerVision: { ch: '计算机视觉', en: 'Computer Vision' },
      nlp: { ch: '自然语言', en: 'NLP' }
    },
    tooltip: {
      drag: {
        ch: '拖拽到画布添加节点',
        en: 'Drag to canvas to add node'
      },
      doubleClick: {
        ch: '双击快速添加节点',
        en: 'Double-click to add quickly'
      },
      favorite: {
        ch: '收藏常用算子',
        en: 'Favorite frequently used operators'
      }
    },
    empty: {
      title: {
        ch: '未找到匹配的算子',
        en: 'No operators found'
      },
      subtitle: {
        ch: '尝试调整搜索条件或职业筛选',
        en: 'Try adjusting search or profession filter'
      }
    },
    instructions: {
      ch: '拖拽或双击均可添加算子',
      en: 'Drag/double-click to add'
    }
  };

  // 改进的职业配色方案
  const professionConfig = {
    all: { name: i18n.profession.all, color: '#667eea', bgColor: 'rgba(102, 126, 234, 0.08)' },
    dataAnalysis: { name: i18n.profession.dataAnalysis, color: '#10b981', bgColor: 'rgba(16, 185, 129, 0.08)' },
    machineLearning: { name: i18n.profession.machineLearning, color: '#f59e0b', bgColor: 'rgba(245, 158, 11, 0.08)' },
    largeModel: { name: i18n.profession.largeModel, color: '#8b5cf6', bgColor: 'rgba(139, 92, 246, 0.08)' },
    agent: { name: i18n.profession.agent, color: '#ef4444', bgColor: 'rgba(239, 68, 68, 0.08)' },
    computerVision: { name: i18n.profession.computerVision, color: '#06b6d4', bgColor: 'rgba(6, 182, 212, 0.08)' },
    nlp: { name: i18n.profession.nlp, color: '#84cc16', bgColor: 'rgba(132, 204, 22, 0.08)' }
  };

  // 职业映射
  const professionMapping = {
    'DataSource(File)': 'dataAnalysis',
    'TableProcess': 'dataAnalysis',
    'DataAnalysis': 'dataAnalysis',
    'TabStructAlgo': 'machineLearning',
    'FlowControl': 'all',
    'ImageProcess': 'computerVision',
    'ComputerVision': 'computerVision',
    'NLP': 'nlp',
    'largeModel': 'largeModel',
    'DistAcceleration': 'machineLearning',
    'OnlineDataSource': 'agent'
  };

  const toggleGroup = (groupName) => {
    if (openGroups.includes(groupName)) {
      setOpenGroups(openGroups.filter(name => name !== groupName));
    } else {
      setOpenGroups([...openGroups, groupName]);
    }
  };

  const toggleItemGroup = (groupName) => {
    setExpandedItemGroups(prev => ({
      ...prev,
      [groupName]: !prev[groupName]
    }));
  };

  const toggleFavorite = (nodeId, event) => {
    event.stopPropagation();
    const newFavorites = new Set(favorites);
    if (newFavorites.has(nodeId)) {
      newFavorites.delete(nodeId);
    } else {
      newFavorites.add(nodeId);
    }
    setFavorites(newFavorites);
  };

  const getLabel = (label) => {
    return displayMode === 'chinese' ? chineseLabels[label] || label : label;
  };

  // 过滤节点
  const filteredGroups = candidateNodes
  .filter(group => {
    const groupProfession = professionMapping[group.group] || 'all';
    if (profession !== 'all' && groupProfession !== profession) return false;

    if (searchText) {
      const filteredNodes = group.nodes.filter(node => {
        // 根据当前语言选择要匹配的标签
        const labelToMatch = language === 'zh'
          ? (chineseLabels[node.data.label] || node.data.label)  // 中文用中文标签
          : node.data.label;  // 英文用原标签

        return labelToMatch.toLowerCase().includes(searchText.toLowerCase());
      });
      return filteredNodes.length > 0;
    }
    return true;
  })
  .map(group => ({
    ...group,
    nodes: group.nodes.filter(node => {
      if (searchText) {
        const labelToMatch = language === 'zh'
          ? (chineseLabels[node.data.label] || node.data.label)
          : node.data.label;
        return labelToMatch.toLowerCase().includes(searchText.toLowerCase());
      }
      return true;
    })
  }))
  .filter(group => group.nodes.length > 0);

   const handleCardClick = (node) => (e) => {
    const now = Date.now();
    const DOUBLE_CLICK_THRESHOLD = 300; // 300ms 内两次点击算双击

    if (lastClickTimeRef.current && (now - lastClickTimeRef.current) < DOUBLE_CLICK_THRESHOLD) {
      // 是双击！
      e.preventDefault();
      e.stopPropagation();

      console.log('🎯 Double click confirmed:', node.data.label);

      // 正确调用父组件的回调，并传入 node
      if (onDoubleClick) {
        onDoubleClick(node); // 直接传 node，不要 event
      }

      // 重置点击时间
      lastClickTimeRef.current = 0;
    } else {
      // 是单击，记录时间
      lastClickTimeRef.current = now;
    }
  };

  const handleDragStart = (event, node) => {
    event.dataTransfer.setData('application/reactflow', JSON.stringify(node));
    if (onDragStart) {
      onDragStart(event, node);
    }
  };

  const getDescription = (node) => {
    if (!node.data.description) return '';
    if (typeof node.data.description === 'string') {
      return node.data.description;
    }
    return node.data.description[language] || node.data.description.ch || '';
  };

  const renderTooltipContent = (node,group="agent") => {
    const groupProfession = professionMapping[node.group] || 'all';
    const professionInfo = professionConfig[groupProfession];
    const description = getDescription(node);
    return (
      <div style={{ maxWidth: '280px', fontSize: '13px', lineHeight: '1.4' }}>
        <div style={{ fontWeight: 600, marginBottom: '6px', fontSize: '14px' }}>
          <span style={{ color: professionInfo.color,marginLeft: '0px', fontSize: '13px',fontWeight: 600, fontWeight: 'Blod' }}>
             {node.data.label}
          </span>
        </div>
        {description && (
          <div style={{
            color: '#475569',
            marginBottom: '8px',
            borderLeft: '3px solid #3b82f6',
            paddingLeft: '8px'
          }}>
            {description}
          </div>
        )}
        <div style={{
          fontSize: '11px',
          color: '#94a3b8',
          borderTop: '1px solid #f1f5f9',
          paddingTop: '6px'
        }}>
          {language === 'ch' ? '双击或拖拽添加到画布' : 'Double-click or drag to canvas'}
        </div>
      </div>
    );
  };

  // 获取要显示的节点（处理折叠逻辑）
  const getDisplayNodes = (group) => {
    const MAX_ITEMS = displayMode === 'icon' ? 6 : 2; // 图标模式下显示6个（2行），列表模式下显示2个
    const isTableProcess = group.group === 'TableProcess';

    if (!isTableProcess || group.nodes.length <= MAX_ITEMS || expandedItemGroups[group.group]) {
      return group.nodes;
    }

    return group.nodes.slice(0, MAX_ITEMS);
  };

  return (
    <div style={{
      width: '220px',
      height: '100%',
      background: 'linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)',
      padding: '12px',
      overflowY: 'auto',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
    }}>
      {/* 自定义滚动条 */}
      <style>
        {`
          ::-webkit-scrollbar { width: 4px; }
          ::-webkit-scrollbar-track { background: #f1f5f9; border-radius: 2px; }
          ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #cbd5e1 0%, #94a3b8 100%);
            border-radius: 2px;
          }
          ::-webkit-scrollbar-thumb:hover { background: #64748b; }
        `}
      </style>

      {/* 标题区域 */}
      <div style={{
        textAlign: 'center',
        marginBottom: '16px',
        paddingBottom: '12px',
        borderBottom: '1px solid #e2e8f0'
      }}>
        <h3 style={{
          margin: 0,
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
          fontSize: '16px',
          fontWeight: 700,
          lineHeight: '1.2'
        }}>
          {i18n.title.en}
          <div style={{ fontSize: '12px', fontWeight: 500, color: '#64748b' }}>

          </div>
        </h3>

      </div>

      {/* 搜索和筛选区域 */}
      <div style={{ marginBottom: '12px' }}>
        <Input
          placeholder={i18n.search.placeholder.ch}
          prefix={<SearchOutlined style={{ color: '#94a3b8', fontSize: '12px' }} />}
          value={searchText}
          onChange={(e) => setSearchText(e.target.value)}
          style={{
            marginBottom: '8px',
            borderRadius: '6px',
            border: '1px solid #e2e8f0',
            fontSize: '12px',
            height: '28px'
          }}
          allowClear
        />

        <Select
          value={profession}
          onChange={setProfession}
          style={{ width: '100%', borderRadius: '6px', fontSize: '12px' }}
          dropdownStyle={{ borderRadius: '6px' }}
          size="small"
        >
          {Object.entries(professionConfig).map(([key, config]) => (
            <Option key={key} value={key}>
              <Space size={4}>
                <span style={{ color: config.color, fontSize: '10px' }}>
                  {/* 图标逻辑保持不变 */}
                  {config.name.ch === '全部' ? <AppstoreOutlined /> :
                   config.name.ch === '数据分析' ? <LineChartOutlined /> :
                   config.name.ch === '机器学习' ? <DatabaseOutlined /> :
                   config.name.ch === '大模型' ? <RobotOutlined /> :
                   config.name.ch === '智能体' ? <MessageOutlined /> :
                   config.name.ch === '计算机视觉' ? <PictureOutlined /> :
                   <CodeOutlined />}
                </span>
                {/* 根据语言状态显示对应文本 */}
                <span style={{ fontSize: '14px' }}>
                  {language === 'zh' ? config.name.ch : config.name.en}
                </span>
              </Space>
            </Option>
          ))}
        </Select>
      </div>

      {/* 操作提示 */}
      <div style={{
        background: 'rgba(59, 130, 246, 0.05)',
        border: '1px solid rgba(59, 130, 246, 0.1)',
        borderRadius: '6px',
        padding: '6px 8px',
        marginBottom: '12px',
        fontSize: '10px',
        color: '#3b82f6',
        lineHeight: '1.3'
      }}>
        <br />
        <span style={{ fontSize: '12px', color: '#64748b' }}>
          💡{i18n.instructions.en}
        </span>
      </div>

      {/* 显示模式切换 */}
      <div style={{
        display: 'flex',
        gap: '1px',
        marginBottom: '16px',
        background: 'rgba(255, 255, 255, 0.6)',
        padding: '2px',
        borderRadius: '8px',
        backdropFilter: 'blur(8px)'
      }}>
        {[
          { key: 'icon', label: i18n.displayMode.icon, icon: <AppstoreOutlined /> },
          { key: 'list', label: i18n.displayMode.list, icon: <DatabaseOutlined /> },
          { key: 'chinese', label: i18n.displayMode.chinese, icon: <MessageOutlined /> }
        ].map(mode => (
          <Tooltip key={mode.key} title={`${mode.label.en} Mode`}>
            <Button
              type={displayMode === mode.key ? 'primary' : 'text'}
              size="small"
              icon={React.cloneElement(mode.icon, { style: { fontSize: '12px' } })}
              onClick={() => setDisplayMode(mode.key)}
              style={{
                flex: 1,
                borderRadius: '2px',
                border: displayMode === mode.key ? '1px solid #3b82f6' : '1px solid transparent',
                background: displayMode === mode.key ?
                  'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)' : 'transparent',
                color: displayMode === mode.key ? 'white' : '#64748b',
                fontSize: '12px',
                height: '24px',
                padding: '0 4px'
              }}
            >
               {mode.label[language]}
            </Button>
          </Tooltip>
        ))}

        {/* 语言切换按钮 */}
        <Tooltip title={language === 'ch' ? 'Switch to English' : '切换到中文'}>
          <Button
            type="text"
            size="small"
            onClick={() => setLanguage(language === 'ch' ? 'en' : 'ch')}
            style={{
              borderRadius: '2px',
              border: '2px solid #e2e8f0',
              background: 'white',
              color: '#475569',
              fontSize: '11px',
              height: '18px',
              minWidth: '30px'
            }}
          >
            {language === 'ch' ? 'EN' : '中'}
          </Button>
        </Tooltip>
      </div>

      {/* 节点组列表 */}
      <div style={{ gap: '16px', display: 'flex', flexDirection: 'column' }}>
        {filteredGroups.map((group) => {
          const groupProfession = professionMapping[group.group] || 'all';
          const professionInfo = professionConfig[groupProfession];
          const displayNodes = getDisplayNodes(group);
          const isTableProcess = group.group === 'TableProcess';
          const hasMoreItems = isTableProcess && group.nodes.length > (displayMode === 'icon' ? 6 : 2);

          return (
            <div key={group.group} style={{
              background: 'rgba(255, 255, 255, 0.8)',
              borderRadius: '8px',
              padding: '8px',
              backdropFilter: 'blur(8px)',
              border: '1px solid rgba(255, 255, 255, 0.5)',
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.04)'
            }}>
              {/* 组标题 */}
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: '8px',
                  cursor: 'pointer',
                  userSelect: 'none'
                }}
                onClick={() => toggleGroup(group.group)}
              >
                <Space size={4}>
                  <span style={{
                    color: professionInfo.color,
                    fontSize: '12px'
                  }}>
                    {professionInfo.name.ch === '全部' ? <AppstoreOutlined /> :
                     professionInfo.name.ch === '数据分析' ? <LineChartOutlined /> :
                     professionInfo.name.ch === '机器学习' ? <DatabaseOutlined /> :
                     professionInfo.name.ch === '大模型' ? <RobotOutlined /> :
                     professionInfo.name.ch === '智能体' ? <MessageOutlined /> :
                     professionInfo.name.ch === '计算机视觉' ? <PictureOutlined /> :
                     <CodeOutlined />}
                  </span>
                  <span style={{
                    fontSize: '16px',
                    fontWeight: 600,
                     color:  `${professionInfo.color}`,
                  }}>
                    {getLabel(group.group)}
                  </span>
                </Space>
                <Button
                  type="text"
                  icon={openGroups.includes(group.group) ?
                    <UpOutlined style={{ fontSize: '12px', color: '#64748b' }} /> :
                    <DownOutlined style={{ fontSize: '12px', color: '#64748b' }} />
                  }
                  style={{ padding: 0, minWidth: 'auto', height: '16px' }}
                />
              </div>

              {/* 节点列表 */}
              {openGroups.includes(group.group) && (
                <div>
                  <div style={{
                    display: displayMode === 'icon' ? 'grid' : 'flex',
                    gridTemplateColumns: 'repeat(3, 1fr)',
                    gap: '4px',
                    flexDirection: 'column'
                  }}>
                    {displayNodes.map((node) => (
                      <Tooltip
                        key={node.id}
                        title={renderTooltipContent(node,group.group)}
                        placement="right"
                        color="white"
                      >
                        <Card
                          onDragStart={(event) => handleDragStart(event, node)}
                          onClick={handleCardClick(node)}
                          draggable
                          size="small"
                          style={{
                            margin: 0,
                            border: '1px solid rgba(226, 232, 240, 0.8)',
                            borderRadius: '6px',
                            background: favorites.has(node.id) ?
                              'linear-gradient(135deg, #fef3c7 0%, #fef7cd 100%)' :
                              'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)',
                            boxShadow: '0 1px 4px rgba(0, 0, 0, 0.03)',
                            cursor: 'grab',
                            transition: 'all 0.2s ease',
                            transform: 'translateY(0)',
                             userSelect: 'none'
                          }}
                          bodyStyle={{
                            padding: displayMode === 'icon' ? '6px 4px' : '4px 6px',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: displayMode === 'icon' ? 'center' : 'flex-start',
                            gap: '4px',
                            position: 'relative',
                            minHeight: displayMode === 'icon' ? '40px' : '32px'
                          }}
                          hoverable
                          onMouseEnter={(e) => {
                            e.currentTarget.style.transform = 'translateY(-1px)';
                            e.currentTarget.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.08)';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.transform = 'translateY(0)';
                            e.currentTarget.style.boxShadow = '0 1px 4px rgba(0, 0, 0, 0.03)';
                          }}
                        >
                          {/* 收藏按钮 */}
                          <div style={{
                            fontSize: displayMode === 'icon' ? '16px' : '12px',
                            color: professionInfo.color,
                            flexShrink: 0
                          }}>
                            {node.data.icon}
                          </div>

                          {displayMode !== 'icon' && (
                            <div style={{
                              fontSize: '14px',
                              fontWeight: 500,
                              lineHeight: '1.2',
                              flex: 1
                            }}>
                              {getLabel(node.data.label)}
                            </div>
                          )}
                        </Card>
                      </Tooltip>
                    ))}
                  </div>

                  {/* 显示更多/收起按钮 */}
                  {hasMoreItems && (
                    <div style={{
                      marginTop: '8px',
                      textAlign: 'center'
                    }}>
                      <Button
                        type="text"
                        size="small"
                        onClick={() => toggleItemGroup(group.group)}
                        style={{
                          fontSize: '11px',
                          color: '#3b82f6',
                          padding: '2px 8px',
                          height: '24px'
                        }}
                        icon={expandedItemGroups[group.group] ?
                          <UpOutlined style={{ fontSize: '10px' }} /> :
                          <DownOutlined style={{ fontSize: '10px' }} />}
                      >
                        {expandedItemGroups[group.group] ?
                          (language === 'ch' ? '收起' : 'Collapse') :
                          `${language === 'ch' ? '显示更多' : 'Show More'} (${group.nodes.length - (displayMode === 'icon' ? 6 : 2)})`}
                      </Button>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* 空状态 */}
      {filteredGroups.length === 0 && (
        <div style={{
          textAlign: 'center',
          padding: '30px 12px',
          color: '#64748b'
        }}>
          <div style={{ fontSize: '32px', marginBottom: '12px' }}>🔍</div>
          <div style={{ fontSize: '13px', fontWeight: 800 }}>{i18n.empty.title.ch}</div>
          <div style={{ fontSize: '11px', marginTop: '2px' }}>{i18n.empty.title.en}</div>
          <div style={{ fontSize: '12px', marginTop: '4px', color: '#94a3b8' }}>
            {i18n.empty.subtitle.ch}
          </div>
          <div style={{ fontSize: '12px', marginTop: '2px', color: '#94a3b8' }}>
            {i18n.empty.subtitle.en}
          </div>
        </div>
      )}
    </div>
  );
}

export default CandidateSidebar;