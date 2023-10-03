#!/bin/bash

mkdir -p compiled images

rm -f ./compiled/*.fst ./images/*.pdf

# ############ Compile source transducers ############
for i in sources/*.txt tests/*.txt; do
	echo "Compiling: $i"
    fstcompile --isymbols=syms.txt --osymbols=syms.txt $i | fstarcsort > compiled/$(basename $i ".txt").fst
done

# ############ CORE OF THE PROJECT  ############
#Create mix2numerical
fstconcat "compiled/mmm2mm.fst" "compiled/quasi_mix2numerical.fst" | fstrmepsilon | fstarcsort > "compiled/mix2numerical.fst"

#Create pt2en/en2pt
fstinvert "compiled/pt_en.fst" > "compiled/en_pt.fst"
fstconcat "compiled/pt_en.fst" "compiled/quasi_mix2numerical.fst" | fstrmepsilon | fstarcsort > "compiled/pt2en.fst"
fstconcat "compiled/en_pt.fst" "compiled/quasi_mix2numerical.fst" | fstrmepsilon | fstarcsort > "compiled/en2pt.fst"

#Create datenum2text
fstconcat "compiled/month.fst" "compiled/slash2empty.fst" | fstrmepsilon > "compiled/monthSpace.fst"
fstconcat "compiled/monthspace.fst" "compiled/day.fst" | fstrmepsilon > "compiled/monthday.fst"
fstconcat "compiled/monthday.fst" "compiled/slash2comma.fst" | fstrmepsilon > "compiled/monthdaycomma.fst"
fstconcat "compiled/monthdaycomma.fst" "compiled/year.fst" | fstrmepsilon | fstarcsort > "compiled/datenum2text.fst"

#Create mix2text
fstcompose "compiled/pt2en.fst" "compiled/mix2numerical.fst" | fstrmepsilon | fstarcsort > "compiled/ptoren2numerical.fst"
fstcompose "compiled/ptoren2numerical.fst" "compiled/datenum2text.fst" | fstrmepsilon | fstarcsort > "compiled/mix2text_end.fst"
fstcompose "compiled/mix2numerical.fst" "compiled/datenum2text.fst" | fstrmepsilon | fstarcsort > "compiled/mix2text_init.fst"
fstunion "compiled/mix2text_init.fst" "compiled/mix2text_end.fst" | fstrmepsilon > "compiled/mix2text.fst"

#Create date2text
fstunion "compiled/mix2text.fst" "compiled/datenum2text.fst" | fstrmepsilon > "compiled/date2text.fst"

# ############ generate PDFs  ############
echo "Starting to generate PDFs"
for i in compiled/*.fst; do
	echo "Creating image: images/$(basename $i '.fst').pdf"
   fstdraw --portrait --isymbols=syms.txt --osymbols=syms.txt $i | dot -Tpdf > images/$(basename $i '.fst').pdf
done



# ############      Testing     ############

#Generate files
echo "\n***********************************************************"
echo "Testing a2 (the output is a transducer: fst and pdf)"
echo "***********************************************************"
for w in compiled/t-a2*.fst; do
    fstcompose $w compiled/mix2numerical.fst | fstshortestpath | fstproject --project_output=true |
                  fstrmepsilon | fsttopsort > compiled/$(basename $w ".fst")-out.fst
done
for w in compiled/t-b4*.fst; do
    fstcompose $w compiled/en2pt.fst | fstshortestpath | fstproject --project_output=true |
                  fstrmepsilon | fsttopsort > compiled/$(basename $w ".fst")-out.fst
done
for w in compiled/t-c8*.fst; do
    fstcompose $w compiled/datenum2text.fst | fstshortestpath | fstproject --project_output=true |
                  fstrmepsilon | fsttopsort > compiled/$(basename $w ".fst")-out.fst
done
for w in compiled/t-d9*.fst; do
    fstcompose $w compiled/mix2text.fst | fstshortestpath | fstproject --project_output=true |
                  fstrmepsilon | fsttopsort > compiled/$(basename $w ".fst")-out.fst
done
for w in compiled/t-d10*.fst; do
    fstcompose $w compiled/date2text.fst | fstshortestpath | fstproject --project_output=true |
                  fstrmepsilon | fsttopsort > compiled/$(basename $w ".fst")-out.fst
done
for i in compiled/t-*-out.fst; do
	echo "Creating image: images/$(basename $i '.fst').pdf"
   fstdraw --portrait --isymbols=syms.txt --osymbols=syms.txt $i | dot -Tpdf > images/$(basename $i '.fst').pdf
done

echo "\nThe end"
