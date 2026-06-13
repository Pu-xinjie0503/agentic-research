import {
  DeleteOutlined,
  ReloadOutlined
} from "@ant-design/icons";
import {
  Button,
  Drawer,
  Empty,
  List,
  Popconfirm,
  Space,
  Tag,
  Typography
} from "antd";
import { useCallback, useEffect, useState } from "react";
import {
  clearMemories,
  deleteMemory,
  listMemories
} from "../lib/api";
import type { MemoryCategory, MemoryRecord } from "../types";

const CATEGORY_LABELS: Record<MemoryCategory, string> = {
  profile: "身份",
  preference: "偏好",
  project: "项目"
};

interface MemoryDrawerProps {
  open: boolean;
  refreshToken: string;
  userId: string;
  onClose: () => void;
}

export function MemoryDrawer({
  open,
  refreshToken,
  userId,
  onClose
}: MemoryDrawerProps) {
  const [memories, setMemories] = useState<MemoryRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [mutatingId, setMutatingId] = useState("");
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const response = await listMemories(userId);
      setMemories(response.memories);
    } catch (refreshError) {
      setError(
        refreshError instanceof Error ? refreshError.message : "长期记忆加载失败"
      );
    } finally {
      setLoading(false);
    }
  }, [userId]);

  useEffect(() => {
    if (open) {
      void refresh();
    }
  }, [open, refresh, refreshToken]);

  async function handleDelete(memoryId: string) {
    setMutatingId(memoryId);
    setError("");
    try {
      await deleteMemory(userId, memoryId);
      await refresh();
    } catch (deleteError) {
      setError(
        deleteError instanceof Error ? deleteError.message : "长期记忆删除失败"
      );
    } finally {
      setMutatingId("");
    }
  }

  async function handleClear() {
    setMutatingId("__all__");
    setError("");
    try {
      await clearMemories(userId);
      await refresh();
    } catch (clearError) {
      setError(
        clearError instanceof Error ? clearError.message : "长期记忆清空失败"
      );
    } finally {
      setMutatingId("");
    }
  }

  return (
    <Drawer
      className="memory-drawer"
      destroyOnHidden
      extra={
        <Space>
          <Button
            icon={<ReloadOutlined />}
            loading={loading}
            onClick={() => void refresh()}
          >
            刷新
          </Button>
          <Popconfirm
            description="该操作会软删除当前用户的全部长期记忆。"
            disabled={memories.length === 0}
            okText="清空"
            cancelText="取消"
            title="清空全部记忆？"
            onConfirm={() => void handleClear()}
          >
            <Button
              danger
              disabled={memories.length === 0}
              loading={mutatingId === "__all__"}
            >
              全部清空
            </Button>
          </Popconfirm>
        </Space>
      }
      open={open}
      title={`长期记忆 · ${memories.length}`}
      width={520}
      onClose={onClose}
    >
      <Typography.Paragraph className="memory-drawer-tip">
        仅保存稳定身份、偏好和项目约束。当前请求始终高于历史记忆。
      </Typography.Paragraph>
      {error ? (
        <Typography.Paragraph type="danger">{error}</Typography.Paragraph>
      ) : null}
      <List
        dataSource={memories}
        locale={{ emptyText: <Empty description="暂无长期记忆" /> }}
        loading={loading}
        renderItem={(memory) => (
          <List.Item
            actions={[
              <Popconfirm
                key="delete"
                okText="删除"
                cancelText="取消"
                title="删除这条记忆？"
                onConfirm={() => void handleDelete(memory.id)}
              >
                <Button
                  danger
                  aria-label={`删除 ${memory.key}`}
                  icon={<DeleteOutlined />}
                  loading={mutatingId === memory.id}
                  type="text"
                />
              </Popconfirm>
            ]}
          >
            <List.Item.Meta
              description={
                <div className="memory-meta">
                  <span>来源线程：{memory.source_thread_id.slice(0, 8) || "未知"}</span>
                  <span>
                    更新：{new Date(memory.updated_at).toLocaleString("zh-CN")}
                  </span>
                </div>
              }
              title={
                <div className="memory-title">
                  <Tag color="cyan">{CATEGORY_LABELS[memory.category]}</Tag>
                  <code>{memory.key}</code>
                </div>
              }
            />
            <Typography.Paragraph className="memory-content">
              {memory.content}
            </Typography.Paragraph>
          </List.Item>
        )}
      />
    </Drawer>
  );
}
