%% sn-jnl.cls  --- Springer Nature compatibility shim
%% --------------------------------------------------------------------------
%% This is a LIGHTWEIGHT, self-contained class that re-implements the subset of
%% the official Springer Nature `sn-jnl' author interface used by this
%% manuscript (\title, \author[*], \affil[*], \fnm, \sur, \orgdiv, \orgname,
%% \orgaddress, \city, \country, \email, \abstract, \keywords, \maketitle,
%% \bmhead, \backmatter).  It is layered on the standard `article' class so the
%% manuscript compiles on any TeX Live installation WITHOUT the official
%% template.
%%
%% TO USE THE OFFICIAL SPRINGER NATURE TEMPLATE:
%%   1. Download the SN LaTeX template from
%%      https://www.springernature.com/gp/authors/campaigns/latex-author-support
%%   2. Replace THIS file with the official `sn-jnl.cls' (and add the
%%      sn-*.bst style, e.g. sn-mathphys.bst).
%%   3. In sn-article.tex change \bibliographystyle{plainnat} to
%%      \bibliographystyle{sn-mathphys}.
%%   The rest of sn-article.tex compiles unchanged.
%% --------------------------------------------------------------------------
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{sn-jnl}[2025/06/21 v0.9 Springer Nature compatibility shim]

%% Pass through / silently ignore SN-specific options (sn-mathphys, pdflatex,
%% Numbered, Reference styles, etc.).
\DeclareOption*{}
\ProcessOptions\relax

\LoadClass[11pt,onecolumn]{article}

\RequirePackage[a4paper,margin=1in]{geometry}
\RequirePackage{xparse}
\RequirePackage{amsmath,amssymb,amsthm}
\RequirePackage{graphicx}
\RequirePackage{booktabs}
\RequirePackage{multirow}
\RequirePackage{array}
\RequirePackage{xcolor}
\RequirePackage{algorithm}
\RequirePackage{algpseudocode}
\RequirePackage[numbers,sort&compress]{natbib}
\RequirePackage[font=small,labelfont=bf]{caption}
\RequirePackage{hyperref}
\hypersetup{colorlinks=true,linkcolor=blue!55!black,citecolor=blue!55!black,
            urlcolor=blue!55!black}

%% --- theorem-like environments (SN provides these) ------------------------
\theoremstyle{definition}
\newtheorem{definition}{Definition}
\newtheorem{example}{Example}
\theoremstyle{plain}
\newtheorem{proposition}{Proposition}

%% --- author / affiliation micro-commands (identity printers) --------------
\NewDocumentCommand{\fnm}{m}{#1}
\NewDocumentCommand{\sur}{m}{#1}
\NewDocumentCommand{\orgdiv}{m}{#1}
\NewDocumentCommand{\orgname}{m}{#1}
\NewDocumentCommand{\orgaddress}{m}{#1}
\NewDocumentCommand{\street}{m}{#1}
\NewDocumentCommand{\city}{m}{#1}
\NewDocumentCommand{\postcode}{m}{#1}
\NewDocumentCommand{\state}{m}{#1}
\NewDocumentCommand{\country}{m}{#1}
\NewDocumentCommand{\email}{m}{%
  \ifx\sn@emaillist\@empty
    \gdef\sn@emaillist{\texttt{#1}}%
  \else
    \g@addto@macro\sn@emaillist{, \texttt{#1}}%
  \fi}

%% --- author / affiliation collectors --------------------------------------
\def\sn@authorlist{}
\def\sn@affillist{}
\def\sn@emaillist{}
\def\sn@abstract{}
\def\sn@keywords{}
\def\sn@corresp{}

\RenewDocumentCommand{\author}{s O{} m}{%
  \IfBooleanTF{#1}{\gdef\sn@corresp{1}\def\sn@sup{\textsuperscript{#2,$*$}}}%
                  {\def\sn@sup{\textsuperscript{#2}}}%
  \ifx\sn@authorlist\@empty
    \protected@edef\sn@authorlist{\unexpanded{#3}\unexpanded\expandafter{\sn@sup}}%
  \else
    \protected@edef\sn@authorlist{\unexpanded\expandafter{\sn@authorlist}, %
      \unexpanded{#3}\unexpanded\expandafter{\sn@sup}}%
  \fi}

\NewDocumentCommand{\affil}{s O{} m}{%
  \ifx\sn@affillist\@empty
    \gdef\sn@affillist{\textsuperscript{#2}#3}%
  \else
    \g@addto@macro\sn@affillist{\par\noindent\textsuperscript{#2}#3}%
  \fi}

\RenewDocumentCommand{\abstract}{m}{\gdef\sn@abstract{#1}}
\NewDocumentCommand{\keywords}{m}{\gdef\sn@keywords{#1}}

%% --- title block ----------------------------------------------------------
\NewDocumentCommand{\bmhead}{m}{\par\medskip\noindent\textbf{#1}\quad\ignorespaces}
\NewDocumentCommand{\backmatter}{}{}
\NewDocumentCommand{\pacs}{O{} m}{}

\renewcommand{\maketitle}{%
  \begin{center}%
    {\LARGE\bfseries\@title\par}%
    \vspace{1.0em}%
    {\large \sn@authorlist \par}%
    \vspace{0.6em}%
    {\small\itshape \sn@affillist \par}%
    \ifx\sn@emaillist\@empty\else
      \vspace{0.4em}{\small \sn@emaillist \par}%
    \fi
    \ifx\sn@corresp\@empty\else
      \vspace{0.3em}{\footnotesize $^{*}$Corresponding author.\par}%
    \fi
  \end{center}%
  \vspace{0.6em}%
  \begin{quotation}\noindent
    {\bfseries Abstract\quad}\sn@abstract\par
    \ifx\sn@keywords\@empty\else
      \vspace{0.5em}\noindent{\bfseries Keywords\quad}\sn@keywords\par
    \fi
  \end{quotation}%
  \vspace{0.8em}%
}

\endinput
