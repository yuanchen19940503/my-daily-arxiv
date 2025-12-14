# README（arXiv Recommender：每日arXiv推荐器）

# arXiv Recommender（基于研究兴趣的每日匹配推荐）


## 中文说明

### 1. 项目简介
每天抓取指定 arXiv 分区的 `/new` 列表（我这里使用了 `gr-qc/new` 与 `astro-ph/new`），将每篇论文的标题/作者/摘要文本与仓库中的 `profile.md`进行匹配，计算语义相似度并排序，生成：
- 当天批次 Top-N 推荐列表（网页展示）
- 当天批次 JSON 归档（永久保存）

匹配方法采用 OpenAI Embeddings（向量化）+ cosine similarity（余弦相似度）。需要在 GitHub Secrets 中配置 `OPENAI_API_KEY`，根据我的测试，每次成本小于0.01美元，充个10美元应该就可以用很久了。

---

### 2. 工作原理（简述）
1) 读取 `profile.md`（设置研究兴趣）
2) 从 arXiv `/new` 列表解析论文条目（title/authors/fulltext）
3) 调用 Embeddings API 将文本编码成向量
4) 计算 profile 向量与每篇论文向量的 cosine 相似度
5) 按分数降序排序，输出 Top-N
6) 生成 `docs/index.html` 与 `docs/data/YYYY-MM-DD.json` 并部署到 GitHub Pages

---

### 3. 功能特性
- **每日批次推荐**：只在 arXiv 有新批次时更新；无新批次则不产生新结果
- **Top-N 排序**：可配置 N（默认 40）
- **去重**：同一 arXiv_id 若同时出现在多个列表（cross-list），合并为一条并合并 source
- **网页展示**：
  - 显示相似度分数、来源分区
  - 作者默认隐藏，按钮一键显示/隐藏
- **永久归档**：每日 JSON 存在 `docs/data/`，随仓库永久保存

---

### 4. 配置与自定义
在 `scripts/reco.py` 中修改搜索的分区，我这里使用了astro-ph和gr-qc：
```python
LIST_PAGES = {
  "gr-qc": "https://arxiv.org/list/gr-qc/new",
  "astro-ph": "https://arxiv.org/list/astro-ph/new",
}
