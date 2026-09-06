"""One-shot bootstrap of the journal manuscript (Annals of Operations
Research) from the OPT 2026 workshop source.  Restructures
paper/opt2026.tex into journal order with proofs inline and the
appendices folded into the body, converts the hand-written bibliography
to refs.bib, and writes paper/aor/aor.tex.

Run once; after hand edits begin in aor.tex do NOT rerun (it overwrites).
    python3 paper/aor/tools/from_opt.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'paper' / 'opt2026.tex'
OUT = ROOT / 'paper' / 'aor'
s = SRC.read_text()

body = s[s.index(r'\begin{document}') + len(r'\begin{document}'):s.index(r'\end{document}')]


def cut(text, start_pat, end_pat, include_end=True):
    """Remove and return text[start:end] where start is the first match of
    start_pat and end the first match of end_pat after it."""
    i = text.index(start_pat)
    j = text.index(end_pat, i) + (len(end_pat) if include_end else 0)
    return text[:i] + text[j:], text[i:j].strip('\n')


# ---- abstract, sections, bibliography -------------------------------------
abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', body, re.S).group(1).strip()
body = body[body.index(r'\section{Introduction}'):]
bib_start = body.index(r'\begin{thebibliography}')
bib_end = body.index(r'\end{thebibliography}') + len(r'\end{thebibliography}')
bib = body[bib_start:bib_end]
body = body[:bib_start] + body[bib_end:]
body, ai_par = cut(body, r'\paragraph{AI assistance.}', '\n\n', include_end=False)

# ---- split main text / appendices -----------------------------------------
app_i = body.index(r'\appendix')
main, app = body[:app_i], body[app_i + len(r'\appendix'):]

# appendix pieces
app, conventions = cut(app, r'\paragraph{Conventions.}', '\n\n', include_end=False)
app, lem_sep = cut(app, r'\begin{lemma}[Separability]', r'\end{proof}')
app, lem_side = cut(app, r'\begin{lemma}[Side transfers', r'\end{lemma}')
app, lem_gsc = cut(app, r'\begin{lemma}[Geometric strong convexity', r'\end{proof}')
app, lem_two = cut(app, r'\begin{lemma}[Two-sided schedule]', r'\end{remark}')
proofs = {}
for key in ['lem:block', 'lem:good', 'thm:linear', 'prop:orth', 'lem:gapid', 'thm:activation']:
    app, proofs[key] = cut(app, r'\begin{proof}[ of ' + ('Lemma' if key.startswith('lem') else 'Theorem' if key.startswith('thm') else 'Proposition') + r'~\ref{' + key + '}]', r'\end{proof}')
assert r'\begin{proof}' not in app[:app.index(r'\section{Exact reductions}')], 'unclaimed proof left in App. A'
app, red = cut(app, r'\section{Exact reductions}\label{app:red}', r'\section{Experimental details}', include_end=False)
app, details = cut(app, r'\section{Experimental details}\label{app:details}', r'\section{Extended related work}', include_end=False)
app, related_ext = cut(app, r'\section{Extended related work}\label{app:related}', r'\section{Numerical verification', include_end=False)
num = app[app.index(r'\section{Numerical verification'):].strip()
left = app[:app.index(r'\section{Numerical verification')].replace(r'\section{Proofs}\label{app:proofs}', '').strip()
if left:
    print('WARNING: leftover appendix text not placed:\n', left[:2000], file=sys.stderr)

# ---- interleave proofs into the analysis section ---------------------------
def after_env(text, label, env, insert):
    i = text.index(r'\label{' + label + '}')
    j = text.index(r'\end{' + env + '}', i) + len(r'\end{' + env + '}')
    return text[:j] + '\n' + insert + text[j:]

main = main.replace(
    'Full proofs and the conventions for degenerate cases are in\nAppendix~\\ref{app:proofs}.',
    conventions)
main = after_env(main, 'lem:block', 'lemma', proofs['lem:block'])
main = after_env(main, 'lem:good', 'lemma', proofs['lem:good'])
# geometric strong convexity before the theorem
i = main.index(r'\begin{theorem}[Global linear convergence]')
main = main[:i] + lem_gsc + '\n\n' + main[i:]
# theorem, explanatory paragraph, proof, then the two-sided lemma
i = main.index(r'\begin{proposition}[Near-orthogonal speed-up]')
main = main[:i] + proofs['thm:linear'] + '\n\n' + lem_two + '\n\n' + main[i:]
main = after_env(main, 'prop:orth', 'proposition', proofs['prop:orth'])
main = after_env(main, 'lem:gapid', 'lemma', proofs['lem:gapid'])
# activation: keep the "In words" sentence after the statement, then the proof
i = main.index('Lemmas~\\ref{lem:block}, \\ref{lem:gapid} and~\\ref{lem:twosided},')
main = main[:i] + proofs['thm:activation'] + '\n\n' + main[i:]
# separability / side lemmas at the end of the Setting section
i = main.index(r'\section{Guarded block transfers}')
main = main[:i] + lem_sep + '\n\n' + lem_side + '\n\n' + main[i:]

# ---- sections in journal order ---------------------------------------------
i_exp = main.index(r'\section{Experiments}')
i_rel = main.index(r'\section{Related work}')
i_con = main.index(r'\section{Conclusion}')
intro_to_analysis = main[:i_exp]
exp = main[i_exp:i_rel]
rel = main[i_rel:i_con]
con = main[i_con:]

red = red.replace(r'\section{Exact reductions}\label{app:red}', r'\section{Exact reductions of the SVM variants}\label{app:red}')
details = details.replace(r'\section{Experimental details}\label{app:details}', r'\subsection{Experimental details}\label{app:details}')
num = num.replace(r'\section{Numerical verification of the analysis}\label{app:num}', r'\subsection{Numerical verification of the analysis}\label{app:num}')
related_ext = related_ext.replace(r'\section{Extended related work}\label{app:related}', '')
rel = rel.replace(r'\section{Related work}', r'\section{Related work}\label{sec:related}')

new_body = '\n\n'.join([intro_to_analysis.rstrip(), red.strip(), exp.rstrip(), details.strip(), num.strip(),
                        rel.rstrip(), related_ext.strip(), con.rstrip()])

# amsthm/sn-jnl: the optional argument replaces the whole header
new_body = new_body.replace(r'\begin{proof}[ of ', r'\begin{proof}[Proof of ')

# ---- appendix references become section references ----------------------
for a, b in [(r'Appendix~\ref{', r'Section~\ref{'),
             (r'(Table~\ref{tab:l2}, appendix)', r'(Table~\ref{tab:l2})'),
             ('instances of this appendix', r'instances of Section~\ref{app:details}'),
             ('are in the\nanonymised supplementary repository.', 'are in the public\nrepository (\\todo{URL}).')]:
    new_body = new_body.replace(a, b)

# ---- citations: author-year in parentheses -------------------------------
new_body = re.sub(r'\\cite(?=[\[{])', r'\\citep', new_body)
# "Kalantari's Triangle Algorithm (TA) ... \citep{kalantari2015}" reads fine;
# \citet is used where the citation is the subject
new_body = new_body.replace(r"The enhanced\nTA (ETA) of\n\citep{gupta2026}", r"The enhanced TA (ETA) of \citet{gupta2026}")
new_body = new_body.replace("The enhanced\nTA (ETA) of\n\\citep{gupta2026}", "The enhanced TA (ETA) of \\citet{gupta2026}")

# ---- bibliography -> BibTeX ------------------------------------------------
entries = {}
for m in re.finditer(r'\\bibitem\{(\w+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})', bib, re.S):
    entries[m.group(1)] = ' '.join(m.group(2).split())
BIB = {
 'kalantari2015': ('article', dict(author='Kalantari, Bahman', title='A characterization theorem and an algorithm for a convex hull problem', journal='Annals of Operations Research', volume='226', number='1', pages='301--349', year='2015')),
 'gupta2026': ('article', dict(author='Gupta, Mayank and Kalantari, Bahman', title='An enhanced triangle algorithm for large-scale support vector machine optimization', journal='Journal of Intelligent Decision Making and Information Science', volume='3', number='9s', year='2026')),
 'lj2015': ('inproceedings', dict(author='Lacoste-Julien, Simon and Jaggi, Martin', title='On the global linear convergence of {Frank--Wolfe} optimization variants', booktitle='Advances in Neural Information Processing Systems', volume='28', year='2015')),
 'mdm1974': ('article', dict(author='Mitchell, B. F. and Demyanov, V. F. and Malozemov, V. N.', title='Finding the point of a polyhedron closest to the origin', journal='SIAM Journal on Control', volume='12', number='1', pages='19--26', year='1974')),
 'gilbert1966': ('article', dict(author='Gilbert, Elmer G.', title='An iterative procedure for computing the minimum of a quadratic form on a convex set', journal='SIAM Journal on Control', volume='4', number='1', pages='61--80', year='1966')),
 'keerthi2000': ('article', dict(author='Keerthi, S. Sathiya and Shevade, Shirish K. and Bhattacharyya, Chiranjib and Murthy, K. R. K.', title='A fast iterative nearest point algorithm for support vector machine classifier design', journal='IEEE Transactions on Neural Networks', volume='11', number='1', pages='124--136', year='2000')),
 'guelat1986': ('article', dict(author='Gu{\\\'e}lat, Jacques and Marcotte, Patrice', title="Some comments on {Wolfe}'s `away step'", journal='Mathematical Programming', volume='35', number='1', pages='110--119', year='1986')),
 'bennett2000duality': ('inproceedings', dict(author='Bennett, Kristin P. and Bredensteiner, Erin J.', title='Duality and geometry in {SVM} classifiers', booktitle='Proceedings of the 17th International Conference on Machine Learning (ICML)', year='2000')),
 'lacoste2013bcfw': ('inproceedings', dict(author='Lacoste-Julien, Simon and Jaggi, Martin and Schmidt, Mark and Pletscher, Patrick', title='Block-coordinate {Frank--Wolfe} optimization for structural {SVMs}', booktitle='Proceedings of the 30th International Conference on Machine Learning (ICML)', year='2013')),
 'elghaoui2012': ('article', dict(author='El Ghaoui, Laurent and Viallon, Vivian and Rabbani, Tarek', title='Safe feature elimination in sparse supervised learning', journal='Pacific Journal of Optimization', volume='8', number='4', pages='667--698', year='2012')),
 'ogawa2013': ('inproceedings', dict(author='Ogawa, Kohei and Suzuki, Yoshiki and Takeuchi, Ichiro', title='Safe screening of non-support vectors in pathwise {SVM} computation', booktitle='Proceedings of the 30th International Conference on Machine Learning (ICML)', year='2013')),
 'tsuji2022': ('inproceedings', dict(author='Tsuji, Kazuma K. and Tanaka, Ken\'ichiro and Pokutta, Sebastian', title='Pairwise conditional gradients without swap steps and sparser kernel herding', booktitle='Proceedings of the 39th International Conference on Machine Learning (ICML)', year='2022')),
 'combettes2020': ('inproceedings', dict(author='Combettes, Cyrille W. and Pokutta, Sebastian', title='Boosting {Frank--Wolfe} by chasing gradients', booktitle='Proceedings of the 37th International Conference on Machine Learning (ICML)', year='2020')),
 'rinaldi2023': ('article', dict(author='Rinaldi, Francesco and Zeffiro, Damiano', title='Avoiding bad steps in {Frank--Wolfe} variants', journal='Computational Optimization and Applications', volume='84', number='1', pages='225--264', year='2023')),
 'fercoq2015': ('inproceedings', dict(author='Fercoq, Olivier and Gramfort, Alexandre and Salmon, Joseph', title='Mind the duality gap: safer rules for the {Lasso}', booktitle='Proceedings of the 32nd International Conference on Machine Learning (ICML)', year='2015')),
 'wang2014': ('inproceedings', dict(author='Wang, Jie and Wonka, Peter and Ye, Jieping', title='Scaling {SVM} and least absolute deviations via exact data reduction', booktitle='Proceedings of the 31st International Conference on Machine Learning (ICML)', year='2014')),
 'ndiaye2017': ('article', dict(author='Ndiaye, Eugene and Fercoq, Olivier and Gramfort, Alexandre and Salmon, Joseph', title='Gap safe screening rules for sparsity enforcing penalties', journal='Journal of Machine Learning Research', volume='18', number='128', pages='1--33', year='2017')),
 'shibagaki2016': ('inproceedings', dict(author='Shibagaki, Atsushi and Karasuyama, Masayuki and Hatano, Kohei and Takeuchi, Ichiro', title='Simultaneous safe screening of features and samples in doubly sparse modeling', booktitle='Proceedings of the 33rd International Conference on Machine Learning (ICML)', year='2016')),
 'bomze2020': ('article', dict(author='Bomze, Immanuel M. and Rinaldi, Francesco and Zeffiro, Damiano', title='Active set complexity of the away-step {Frank--Wolfe} algorithm', journal='SIAM Journal on Optimization', volume='30', number='3', pages='2470--2500', year='2020')),
 'garber2020': ('inproceedings', dict(author='Garber, Dan', title='Revisiting {Frank--Wolfe} for polytopes: strict complementarity and sparsity', booktitle='Advances in Neural Information Processing Systems', volume='33', year='2020')),
 'gartner2009': ('inproceedings', dict(author='G{\\"a}rtner, Bernd and Jaggi, Martin', title='Coresets for polytope distance', booktitle='Proceedings of the 25th Annual Symposium on Computational Geometry (SoCG)', year='2009')),
 'clarkson2010': ('article', dict(author='Clarkson, Kenneth L.', title='Coresets, sparse greedy approximation, and the {Frank--Wolfe} algorithm', journal='ACM Transactions on Algorithms', volume='6', number='4', pages='63', year='2010')),
 'tsang2005': ('article', dict(author='Tsang, Ivor W. and Kwok, James T. and Cheung, Pak-Ming', title='Core vector machines: fast {SVM} training on very large data sets', journal='Journal of Machine Learning Research', volume='6', pages='363--392', year='2005')),
 'nanculef2014': ('article', dict(author='{\\~N}anculef, Ricardo and Frandi, Emanuele and Sartori, Claudio and Allende, H{\\\'e}ctor', title='A novel {Frank--Wolfe} algorithm. Analysis and applications to large-scale {SVM} training', journal='Information Sciences', volume='285', pages='66--99', year='2014')),
 'lopez2015': ('article', dict(author='L{\\\'o}pez, Jorge and Dorronsoro, Jos{\\\'e} R.', title='Linear convergence rate for the {MDM} algorithm for the nearest point problem', journal='Pattern Recognition', volume='48', number='4', pages='1510--1522', year='2015')),
 'wolfe1976': ('article', dict(author='Wolfe, Philip', title='Finding the nearest point in a polytope', journal='Mathematical Programming', volume='11', number='1', pages='128--149', year='1976')),
 'pena2019': ('article', dict(author='Pe{\\~n}a, Javier and Rodr{\\\'\\i}guez, Daniel', title='Polytope conditioning and linear convergence of the {Frank--Wolfe} algorithm', journal='Mathematics of Operations Research', volume='44', number='1', pages='1--18', year='2019')),
}
missing = set(entries) - set(BIB)
assert not missing, missing
lines = ['% Generated from paper/opt2026.tex by paper/aor/tools/from_opt.py.',
         '% TODO (AOR, APA 7): add doi = {https://doi.org/...} to every entry that has one.', '']
for key, (kind, f) in BIB.items():
    lines.append(f'@{kind}{{{key},')
    for k, v in f.items():
        lines.append(f'  {k} = {{{v}}},')
    lines.append('}\n')
(OUT / 'refs.bib').write_text('\n'.join(lines))

# ---- wrapper ---------------------------------------------------------------
preamble_cmds = s[s.index(r'\newcommand{\conv}'):s.index(r'\title[')]
wrapper = r"""% Journal version for Annals of Operations Research (Springer Nature).
% Bootstrapped from ../opt2026.tex by tools/from_opt.py; hand-edited after.
%
% Uses the Springer Nature class sn-jnl.cls when it is present in this
% directory (download the "Journal article template package" from
% https://www.springernature.com/gp/authors/campaigns/latex-author-support
% and copy sn-jnl.cls and the sn-*.bst files here); otherwise falls back to
% article so that the draft builds while the content is being written.
\IfFileExists{sn-jnl.cls}{%
  \documentclass[pdflatex,sn-apa]{sn-jnl}%  AOR: APA 7 references
  \newif\ifsnjnl\snjnltrue
}{%
  \documentclass[11pt,a4paper]{article}
  \newif\ifsnjnl\snjnlfalse
  \usepackage[margin=2.5cm]{geometry}
  \usepackage{amsmath,amssymb,amsthm}
  \usepackage[authoryear,round]{natbib}
  \usepackage{graphicx}
  \usepackage{hyperref}
  \newtheorem{theorem}{Theorem}
  \newtheorem{lemma}[theorem]{Lemma}
  \newtheorem{proposition}[theorem]{Proposition}
  \newtheorem{corollary}[theorem]{Corollary}
  \newtheorem{remark}[theorem]{Remark}
}
\ifsnjnl
  \theoremstyle{thmstyleone}
  \newtheorem{theorem}{Theorem}
  \newtheorem{lemma}[theorem]{Lemma}
  \newtheorem{proposition}[theorem]{Proposition}
  \newtheorem{corollary}[theorem]{Corollary}
  \theoremstyle{thmstyletwo}
  \newtheorem{remark}[theorem]{Remark}
\fi
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{mathtools}
\usepackage{booktabs}
\usepackage[expansion=false]{microtype}
\usepackage{xcolor}
\usepackage[ruled,linesnumbered]{algorithm2e}
% the OPT class wrapped algorithm2e's float as "algorithm2e"; keep the name
\newenvironment{algorithm2e}[1][t]{\begin{algorithm}[#1]}{\end{algorithm}}
\hypersetup{hidelinks}
\graphicspath{{../}{../figs/}}
\setlength{\emergencystretch}{2em}
\newcommand{\todo}[1]{\textcolor{red}{[TODO: #1]}}

""" + preamble_cmds + r"""
\begin{document}

\ifsnjnl
\title[Guarded block transfers for polytope distance]{Guarded block transfers for polytope distance: linear convergence with explicit constants and gap-certified screening for SVM training}
\author*[1]{\fnm{Mayank} \sur{Gupta}}\email{iammayankg@gmail.com}
\affil*[1]{\orgname{\todo{affiliation}}, \orgaddress{\city{\todo{city}}, \country{\todo{country}}}}
\abstract{""" + abstract + r"""}
\keywords{polytope distance, Frank--Wolfe methods, Triangle Algorithm, support vector machines, safe screening, linear convergence}
\maketitle
\else
\title{Guarded block transfers for polytope distance: linear convergence with explicit constants and gap-certified screening for SVM training}
\author{Mayank Gupta\thanks{\todo{affiliation}; \texttt{iammayankg@gmail.com}}}
\date{Draft of \today}
\maketitle
\begin{abstract}
""" + abstract + r"""
\end{abstract}
\noindent\textbf{Keywords:} polytope distance; Frank--Wolfe methods; Triangle Algorithm; support vector machines; safe screening; linear convergence
\fi

""" + new_body + r"""

\ifsnjnl\backmatter\fi

\section*{Declarations}
\paragraph{Funding.} \todo{No funding was received for this work, or name the funder.}
\paragraph{Competing interests.} \todo{The author has no competing interests to declare, or state them.}
\paragraph{Data availability.} The datasets analysed are the public LIBSVM
benchmark sets a9a, w8a, ijcnn1, gisette and covtype
(\url{https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/}); the
synthetic instances are generated by the released code from fixed seeds.
\paragraph{Code availability.} \todo{public repository URL and archived
release DOI (e.g.\ Zenodo)}; every table and figure is produced by a
script named in Section~\ref{app:details}.
\paragraph{Use of AI tools.} """ + ai_par.replace(r'\paragraph{AI assistance.}', '').strip() + r"""
\paragraph{Prior publication.} A six-page version of this work was
presented at the OPT 2026 workshop (NeurIPS 2026, non-archival); this
article extends it with \todo{list: full proofs, the exact reductions,
the isolated timing study, the n-scaling experiment, ...}.

\ifsnjnl\else\bibliographystyle{plainnat}\fi
\bibliography{refs}

\end{document}
"""
(OUT / 'aor.tex').write_text(wrapper)
print('wrote', OUT / 'aor.tex', 'and refs.bib')
# report leftovers to fix by hand
txt = wrapper
for pat in ['Appendix', 'appendix', 'anonymised', 'supplementary', 'workshop', 'page', r'\ref{app:proofs}']:
    for m in re.finditer(pat, txt):
        ln = txt[:m.start()].count('\n') + 1
        ctx = txt[max(0, m.start() - 60):m.end() + 60].replace('\n', ' ')
        print(f'{pat:14s} line {ln}: ...{ctx}...')
