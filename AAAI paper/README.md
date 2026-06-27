# AAAI Paper Submission

本目录包含 AAAI 投稿版本的完整 LaTeX 论文。

## 文件说明

- `main.tex`：完整论文（单文件，符合 AAAI 格式要求）
- `aaai.sty` / `aaai.bst` / `fixbib.sty`：AAAI 官方样式文件
- `figures/`：论文所需图片
  - `router_heatmap_test_closed.png`
  - `baseline_comparison.png`
  - `ablation_comparison_test_closed.png`
  - 其他可视化图片

## 编译方式

使用 pdflatex 编译：

```bash
pdflatex main.tex
pdflatex main.tex
```

注意：
- AAAI 要求单 `.tex` 文件，因此所有表格都已内联在 `main.tex` 中
- 不要直接使用 `\input{}` 引入其他 `.tex` 文件
- 最终提交前需删除作者信息（当前为 Anonymous）

## 当前状态

- 论文已包含：摘要、引言、相关工作、方法、实验（主结果/baseline/消融/可视化）、结论
- 待补充：Top-k ablation 结果、GNN 改进实验结果、多随机种子结果
- 待优化：架构图可替换为更精美的 TikZ/PDF 矢量图

## 页数

当前约 5-6 页，AAAI 通常限制 7 页正文 + 1 页引用。
