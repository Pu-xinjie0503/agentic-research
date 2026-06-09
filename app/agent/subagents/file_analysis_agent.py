"""
文件分析子智能体配置模块

将 app/prompt/prompts.yml 中的 file_analysis 配置与上传文件读取工具组装成
DeepAgents 可识别的字典式子智能体。主智能体后续会根据 description
决定是否把用户上传附件、临时文档和表格分析任务分派给它。
"""

from app.agent.prompts import sub_agents_content
from app.tools.upload_file_read_tool import read_file_content

# 文件分析助手只负责读取和分析当前会话工作目录中的上传附件
# 它不负责最终 Markdown/PDF 交付，最终文档生成仍由主智能体掌握
file_analysis_agent = {
    "name": sub_agents_content["file_analysis"]["name"],
    "description": sub_agents_content["file_analysis"]["description"],
    "system_prompt": sub_agents_content["file_analysis"]["system_prompt"],
    "tools": [read_file_content],
}
