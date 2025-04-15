while read entry; do
	echo "repscript: $entry"
	python $1 $entry
done < $2
