#!/usr/bin/env bash

set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"
MVTec_ROOT="${MVTec_ROOT:-/mnt/d/Bootcamp/Project/mvtec_anomaly_detection}"
MVTec_SPLIT="${MVTec_SPLIT:-test/good}"

echo "Prediction test event sender"
echo "API_URL=${API_URL}"
echo "MVTec_ROOT=${MVTec_ROOT}"
echo "MVTec_SPLIT=${MVTec_SPLIT}"
echo

read -r -p "Category (example: bottle): " category
if [[ -z "${category}" ]]; then
  echo "Category is required."
  exit 1
fi

image_dir="${MVTec_ROOT}/${category}/${MVTec_SPLIT}"
if [[ ! -d "${image_dir}" ]]; then
  echo "Directory not found: ${image_dir}"
  echo "Set MVTec_ROOT and/or MVTec_SPLIT env vars if your dataset is elsewhere."
  exit 1
fi

mapfile -t images < <(find "${image_dir}" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | sort)
if [[ "${#images[@]}" -eq 0 ]]; then
  echo "No images found in: ${image_dir}"
  exit 1
fi

echo "Found ${#images[@]} images in ${image_dir}"
echo "Example indexes:"
for idx in 1 2 3; do
  if (( idx <= ${#images[@]} )); then
    printf "  %d) %s\n" "${idx}" "$(basename "${images[$((idx - 1))]}")"
  fi
done
echo


read -r -p "Image number (1-${#images[@]}): " image_number
if ! [[ "${image_number}" =~ ^[0-9]+$ ]]; then
  echo "Image number must be numeric."
  exit 1
fi
if (( image_number < 1 || image_number > ${#images[@]} )); then
 echo "Image number out of range."
 exit 1
fi


image_path="${images[$((image_number - 1))]}"
echo
echo "Sending prediction request:"
echo "  category=${category}"
echo "  image=$(basename "${image_path}")"
echo "  endpoint=${API_URL}/predict/${category}"
echo

curl -sS -X POST "${API_URL}/predict/${category}" -F "image=@${image_path}"
echo
