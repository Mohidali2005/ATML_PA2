B=28100043_PA2_report
pdflatex -interaction=nonstopmode $B >/dev/null 2>&1
bibtex $B >/dev/null 2>&1
pdflatex -interaction=nonstopmode $B >/dev/null 2>&1
pdflatex -interaction=nonstopmode $B >/dev/null 2>&1
grep -n "Overfull\|Undefined\|undefined\|Citation.*undefined" $B.log | head
pdfinfo $B.pdf | grep Pages
for i in $(seq 1 16); do n=$(pdftotext -f $i -l $i -layout $B.pdf - 2>/dev/null | grep -c .); echo -n "p$i:$n "; done; echo
pdftotext -layout $B.pdf - | grep -n "^ *References\|^ *Appendix\|^ *[0-9] *Conclusion"
