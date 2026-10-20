# AI 审核：先在自己的 fork 试用

这版自动分析 Issue、PR 代码和已有 CI 结果，输出中文建议。
不会自动改代码、批准、合并或关闭，也不会执行投稿代码或模型生成的验证代码。
它能帮助筛选和发现问题，但不能代替独立复现与实验。

## 开启步骤

1. 将本分支合并到 **Rette66/kaiwu-pytorch-plugin 的默认分支**，不要先提交到上游仓库。
2. 打开 fork 的 **Settings → Secrets and variables → Actions → Secrets**，
   新建 `OPENAI_API_KEY`，填入你自己的 OpenAI API 项目密钥。
   不要把密钥写在文件、评论或聊天中。代码会发送给 OpenAI 进行分析。
3. 在同一页面的 **Variables** 中添加 `AI_REVIEW_ENABLED`，值填 `true`。
   默认使用 `gpt-5.4`；如需使用账户内其他支持结构化输出的模型，
   添加 `OPENAI_REVIEW_MODEL`。在 OpenAI 项目侧设置用量提醒及可用的预算限制。
4. 若 fork 的 Actions 尚未启用，打开 **Actions** 页面按提示启用。
5. 在 **Actions → AI review → Run workflow** 选择默认分支，先手动试跑：

   | number | repository | 查看什么 |
   | --- | --- | --- |
   | `207` | `qboson/kaiwu-pytorch-plugin` | 是否识别防误删的价值，并说明未实际运行 |
   | `254` | `qboson/kaiwu-pytorch-plugin` | 是否核对求解方向，并指出需要独立公式验证 |
   | `389` | `qboson/kaiwu-pytorch-plugin` | 是否发现 base 中已有相同修复；不要只信 PR 描述 |

   上游 PR 的结果只显示在 fork 的运行摘要和报告附件中，**不会向上游写评论**。
   `repository` 留空则审核 fork 自己的同号 Issue/PR。

## 自动运行

- fork 维护者、成员或协作者新建、编辑或重新打开的非机器人 Issue 自动触发。
  外部用户的 Issue 由维护者在 Actions 页面手动选择编号，避免陌生人反复编辑耗费额度。
- fork 的现有 `CI` 工作流完成后，自动审核对应的最新非草稿 PR。
  外部投稿若等待 GitHub 的首次运行批准，需要先按 GitHub 的要求批准 CI。
- fork 收不到上游 Issue/PR 的事件，当前通过手动选择上游编号试审，
  **不会自动扫描上游积压列表**。
- fork 自己的审核结果更新同一条 `github-actions[bot]` 评论，避免刷屏。
  评论发布前会重新核对标题、描述、状态以及 PR 的 head/base 版本；过期结果不发布。

## 审核范围与限制

审核依据包括：最多 8 个变更文件的 base/head 原文与修改片段、项目说明、
同标题关键词的相关 Issue/PR，以及已有 CI 状态。Issue 中明确给出的仓库路径也会读取。
源文件最多读取 5000 字符，较大的文件优先读取修改位置附近的原文；
修改片段最多 5000 字符；总输入最多 90000 字符。
超出范围、内容不可用、没有检查结果、API 失败或回答不完整时，不能给出通过结论。
大 PR 和没有源代码位置的 Issue 通常会被要求补证据。

每条代码问题必须提供实际读取文件中的逐字引用，引用不存在则本次审核失败。
这能限制虚构引用，**不能证明 AI 对代码的解释一定正确**。
重复检索只是标题关键词搜索，不是完整去重，也没有自动读取论文或运行硬件实验。
现有 CI 成功也不表示作者测试可信、所有测试均执行或模型效果成立。

每次最多调用一次 OpenAI，不自动重试，最多输出 6000 token，工作流最多运行 10 分钟。
快速更新会取消同一来源的旧任务，但已经发出的 API 请求仍可能计费。
这限制单次资源，不是账户级费用上限；仍需注意协作者和 PR 触发量及项目用量。

## 权限和安全

审核代码、规则和工作流取自 fork 默认分支，外部 PR 只作为 API 返回的数据读取。
不会加载 PR 修改的规则，也不会安装 PR 的依赖或调用其脚本。
分析任务只有读取 GitHub 的权限，密钥仅提供给调用 OpenAI 的步骤。
另一个没有 OpenAI 密钥的任务负责更新 fork 评论。原仓库中的普通测试继续独立执行。

停用时，将 `AI_REVIEW_ENABLED` 改为 `false`，并取消正在运行的任务。
无需删除 Issue、PR 或改动模型代码。

## 本地检查

不需要 OpenAI 密钥、PyTorch 或 Kaiwu：

```bash
python3 -m unittest discover -s tests -p test_ai_review.py -v
python3 -m pylint .github/ai_review.py tests/test_ai_review.py
```

只收集公开上游 PR 的真实材料，不调用 AI、不写评论：

```bash
GITHUB_REPOSITORY=Rette66/kaiwu-pytorch-plugin \
GITHUB_EVENT_NAME=workflow_dispatch \
GITHUB_EVENT_PATH=/tmp/ai-review-event.json \
python3 .github/ai_review.py collect
```

其中 `/tmp/ai-review-event.json` 的内容为：

```json
{"inputs": {"number": "207", "repository": "qboson/kaiwu-pytorch-plugin"}}
```

公开读取可以不设置 `GITHUB_TOKEN`，但会受到 GitHub 未登录请求的频率限制。
`review-packet.json` 和 `review.md` 是运行产物，请在临时目录或独立检出中运行上述命令。
