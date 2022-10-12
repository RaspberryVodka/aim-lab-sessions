root_dir="$1"
csv_file="$root_dir/mnist_dataset.csv"
re='^[0-9]+$'


echo "image_path, label" > "$csv_file"
while read -r label; do 
  if [[ $label =~ $re ]] ; then
    while read -r image; do
      echo "$root_dir/$label/$image, $label" >> "$csv_file"
    done < <(ls "$root_dir/$label")
    echo "$label: $(ls "$root_dir/$label" | wc -l)"
  fi
  
done < <(ls "$root_dir")
