#!/usr/bin/env bash
set -euo pipefail

force_metal=0
case "${1:-}" in
  "") ;;
  --force-metal) force_metal=1 ;;
  *)
    printf 'Usage: %s [--force-metal]\n' "$0" >&2
    exit 2
    ;;
esac

cd "$(dirname "$0")"
swift build -c release

bin_dir="$(swift build -c release --show-bin-path)"
shader_dir="$PWD/.build/checkouts/mlx-swift/Source/Cmlx/mlx-generated/metal"
metallib="$bin_dir/mlx.metallib"
if [[ ! -d "$shader_dir" ]]; then
  printf 'MLX Metal shaders are missing: %s\n' "$shader_dir" >&2
  exit 1
fi

if [[ "$force_metal" -eq 0 && -f "$metallib" ]] &&
   [[ -z "$(find "$shader_dir" -type f \( -name '*.metal' -o -name '*.h' \) -newer "$metallib" -print -quit)" ]]; then
  printf 'Using %s\n' "$metallib"
  exit 0
fi

air_dir="$bin_dir/mlx-air"
mkdir -p "$air_dir"
air_files=()
while IFS= read -r -d '' source; do
  name="$(basename "${source%.metal}")"
  air="$air_dir/$name.air"
  xcrun -sdk macosx metal -x metal -Wall -Wextra -fno-fast-math \
    -Wno-c++17-extensions -Wno-c++20-extensions \
    -mmacosx-version-min=14.0 -I"$shader_dir" -c "$source" -o "$air"
  air_files+=("$air")
done < <(find "$shader_dir" -type f -name '*.metal' -print0)

if [[ "${#air_files[@]}" -eq 0 ]]; then
  printf 'MLX Metal shader sources are missing: %s\n' "$shader_dir" >&2
  exit 1
fi

xcrun -sdk macosx metallib "${air_files[@]}" -o "$metallib.new"
mv "$metallib.new" "$metallib"
printf 'Built %s from %s Metal sources\n' "$metallib" "${#air_files[@]}"
