# Blog post: "Training SVMs by Measuring the Distance Between Two Shapes"

`post.md` is the article; `figs/` holds the ten figures in reading order.

Publishing on Medium: Medium does not import local images from Markdown.
Paste the text (Medium's editor accepts Markdown-style headings and
emphasis when pasted, or use its "Import a story" with a hosted copy),
then upload each figure at the marked position - the alt-text in each
`![...]` line names the figure. Figures 8 and 9 are the paper's
consolidated-benchmark and real-data-trace figures; regenerate figure 8
with `python3 src/final_benchmark_fig.py --json <final_benchmark json>`
after `src/final_benchmark.py`, and figure 9 with the trace script. Figure 10 is the regime battery (Table 1 of the paper);
regenerate with `python3 src/regime_fig.py` after `regime_battery.py`.
