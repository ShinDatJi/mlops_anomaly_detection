#!/usr/bin/env bash

set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"
MVTec_ROOT="${MVTec_ROOT:-/mnt/d/Bootcamp/Project/mvtec_anomaly_detection}"
MVTec_TEST_DIR="${MVTec_TEST_DIR:-test}"
TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

echo "Prediction test event sender"
echo "API_URL=${API_URL}"
echo "MVTec_ROOT=${MVTec_ROOT}"
echo "MVTec_TEST_DIR=${MVTec_TEST_DIR}"
echo
echo "Select type test: "
echo "1. Normal image"
echo "2. False category"
echo "3. Invalid image"
echo
read -r TEST_TYPE

case "${TEST_TYPE}" in
1)
  echo "Selected: Normal image"

  if [[ ! -d "${MVTec_ROOT}" ]]; then
    echo "MVTec_ROOT not found: ${MVTec_ROOT}"
    exit 1
  fi

  mapfile -t categories < <(find "${MVTec_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort)
  if [[ "${#categories[@]}" -eq 0 ]]; then
    echo "No categories found in: ${MVTec_ROOT}"
    exit 1
  fi

  echo "Available categories in ${MVTec_ROOT}:"
  for idx in "${!categories[@]}"; do
    printf "  %d) %s\n" "$((idx + 1))" "$(basename "${categories[$idx]}")"
  done
  echo

  read -r -p "Category number (1-${#categories[@]}): " category_number
  if ! [[ "${category_number}" =~ ^[0-9]+$ ]]; then
    echo "Category number must be numeric."
    exit 1
  fi
  if (( category_number < 1 || category_number > ${#categories[@]} )); then
    echo "Category number out of range."
    exit 1
  fi

  category="$(basename "${categories[$((category_number - 1))]}")"

  test_dir="${MVTec_ROOT}/${category}/${MVTec_TEST_DIR}"
  if [[ ! -d "${test_dir}" ]]; then
    echo "Directory not found: ${test_dir}"
    echo "Set MVTec_ROOT and/or MVTec_TEST_DIR env vars if your dataset is elsewhere."
    exit 1
  fi

  mapfile -t folders < <(find "${test_dir}" -mindepth 1 -maxdepth 1 -type d | sort)
  if [[ "${#folders[@]}" -eq 0 ]]; then
    echo "No test subfolders found in: ${test_dir}"
    exit 1
  fi

  echo "Available test subfolders in ${test_dir}:"
  for idx in "${!folders[@]}"; do
    printf "  %d) %s\n" "$((idx + 1))" "$(basename "${folders[$idx]}")"
  done
  echo

  read -r -p "Subfolder number (1-${#folders[@]}): " folder_number
  if ! [[ "${folder_number}" =~ ^[0-9]+$ ]]; then
    echo "Subfolder number must be numeric."
    exit 1
  fi
  if (( folder_number < 1 || folder_number > ${#folders[@]} )); then
    echo "Subfolder number out of range."
    exit 1
  fi

  selected_folder="${folders[$((folder_number - 1))]}"
  mapfile -t images < <(find "${selected_folder}" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | sort)
  if [[ "${#images[@]}" -eq 0 ]]; then
    echo "No images found in: ${selected_folder}"
    exit 1
  fi

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
  echo "Selected subfolder: $(basename "${selected_folder}")"
  echo "Selected file: $(basename "${image_path}")"
  echo
  echo "Sending prediction request:"
  ;;

2) 
  echo "Selected: False category"
  read -r -p "Category (example: non_existent_category): " category
  
  image_path="${TMP_DIR}/dummy_for_invalid_category.png"
  # Any readable file is enough here because category validation happens first.
  printf "not_a_real_png" > "${image_path}"

  echo
  echo "Sending prediction request with false category:"
  ;;

3)
  echo "Selected: Invalid image"
  read -r -p "Category (example: bottle): " category
  if [[ -z "${category}" ]]; then
    echo "Category is required."
    exit 1
  fi  

  image_path="${TMP_DIR}/invalid_image.txt"
  printf "this is not an image" > "${image_path}"
  echo "Created invalid file: ${image_path}"
  
  echo
  echo "Sending prediction request with false image:"
  ;;

esac

echo "  category=${category}"
echo "  image=$(basename "${image_path}")"
echo "  endpoint=${API_URL}/predict/${category}"
echo

read -r -s -p "API key (X-API-Key): " API_KEY
echo

if [[ -z "${API_KEY}" ]]; then
  echo "API key is required. Request cancelled."
  exit 1
fi

response="$(curl -sS -w $'\n%{http_code}' -X POST "${API_URL}/predict/${category}" \
  -H "X-API-Key: ${API_KEY}" \
  -F "image=@${image_path}")"

http_status="${response##*$'\n'}"
response_body="${response%$'\n'*}"

echo "${response_body}"
echo "HTTP status: ${http_status}"
