rm -rf cropped
mkdir cropped
FILES=`ls  -I "*-crop.pdf" `
for f in $FILES
do
  filename=$(basename -- "$f")
  extension="${filename##*.}"
  filename="${filename%.*}"
  #echo $extension

  if [ $extension == 'pdf' ]; then
    echo "Processing $f file..."
    pdfcrop $f
  fi
done
mv *-crop.pdf cropped