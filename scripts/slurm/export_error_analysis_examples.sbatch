#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --job-name=rg_cf_error_imgs
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/home/msai/lius0131/AGI/repostguard-lite
DATA_ROOT=data/raw/community_forensics
OUTPUT_ROOT=reports/assets/error_analysis
MANIFEST="$OUTPUT_ROOT/error_analysis_examples.tsv"
TMP_MANIFEST="$OUTPUT_ROOT/.error_analysis_examples_${SLURM_JOB_ID}.tsv"

cd "$PROJECT_ROOT"
mkdir -p logs "$OUTPUT_ROOT"

command -v convert >/dev/null
command -v identify >/dev/null

CODES=(
    fp_l
    fp_c
    fp_f
    fn_f
    fn_d
    fn_s
    fn_i
)

ROLES=(
    false_positive
    false_positive
    false_positive
    false_negative
    false_negative
    false_negative
    false_negative
)

SAMPLE_IDS=(
    cf_external_unseen_v3_real_laion_26ee8390ffc10966
    cf_test_external_unseen_generator_real_696ae99363416c3a
    cf_test_external_unseen_generator_real_7d9a62f010d3fef7
    cf_test_external_unseen_generator_aigi_e443b5a1c729d1e4
    cf_test_external_unseen_generator_aigi_6b5b17c0a5ce4791
    cf_external_unseen_v3_aigi_other_689deeb7f92c7a13
    cf_external_unseen_v3_aigi_commercial_9b00d2d4e402ab3b
)

SOURCES=(
    LAION
    COCO
    FFHQ
    Firefly_Image2
    Dalle2
    stable_cascade
    Imagen3
)

INPUTS=(
    test_external_unseen_v3_additions/real/cf_external_unseen_v3_real_laion_26ee8390ffc10966_00002615.jpeg.jpg
    test_external_unseen_generator/real/cf_test_external_unseen_generator_real_696ae99363416c3a_000000093076.jpg.jpg
    test_external_unseen_generator/real/cf_test_external_unseen_generator_real_7d9a62f010d3fef7_55660.png.png
    test_external_unseen_generator/aigi/cf_test_external_unseen_generator_aigi_e443b5a1c729d1e4_00000068_1.jpg.jpg
    test_external_unseen_generator/aigi/cf_test_external_unseen_generator_aigi_6b5b17c0a5ce4791_00000550.png.png
    test_external_unseen_v3_additions/aigi/cf_external_unseen_v3_aigi_other_689deeb7f92c7a13_000312.png.png
    test_external_unseen_v3_additions/aigi/cf_external_unseen_v3_aigi_commercial_9b00d2d4e402ab3b_00000184_0.png.png
)

EXPECTED_SHA256=(
    88408f1a8fc9f5ff9a0dc88ffc52182858b509dada39bcfb62fe4d3aaf862042
    1bc6fd389c9b8de694eec8e005b9bfe309167863dc55e21b17c1f804e8940dd6
    55e663c227609999fa0236475ee4b95beacac828c7b47caa277e37f897a00755
    2189d09ec85d6ecd80e373051194674b937c828ca80edccf77303582c209ae90
    101de805757bcc6aa436d84a2e6397746b827116b67ea52e20a622fa15c36f42
    b6f24efd4c23f14855bca45c747ff13d9ac1b30e421cab4a983bc205127413e8
    6deb6fa15ff7c1eaeb2b7df40f95fe964757e3448f0aeba159f3f545a8ee190b
)

OUTPUTS=(
    fp_l_laion.jpg
    fp_c_coco.jpg
    fp_f_ffhq.jpg
    fn_f_firefly_image2.jpg
    fn_d_dalle2.jpg
    fn_s_stable_cascade.jpg
    fn_i_imagen3.jpg
)

cleanup() {
    rm -f "$TMP_MANIFEST"
    for OUTPUT in "${OUTPUTS[@]}"; do
        rm -f "$OUTPUT_ROOT/.${OUTPUT}.${SLURM_JOB_ID}.jpg"
    done
}
trap cleanup EXIT

printf 'code\trole\tsample_id\tsource_or_generator\toriginal_relative_path\toriginal_sha256\tthumbnail_relative_path\tthumbnail_sha256\tdimensions\tbytes\n' > "$TMP_MANIFEST"

for INDEX in "${!CODES[@]}"; do
    INPUT="$DATA_ROOT/${INPUTS[$INDEX]}"
    OUTPUT="$OUTPUT_ROOT/${OUTPUTS[$INDEX]}"
    TMP_OUTPUT="$OUTPUT_ROOT/.${OUTPUTS[$INDEX]}.${SLURM_JOB_ID}.jpg"

    test -s "$INPUT"
    ACTUAL_SHA256=$(sha256sum "$INPUT" | awk '{print $1}')
    test "$ACTUAL_SHA256" = "${EXPECTED_SHA256[$INDEX]}"

    convert "$INPUT" \
        -auto-orient \
        -thumbnail '960x720>' \
        -strip \
        -colorspace sRGB \
        -sampling-factor 4:2:0 \
        -interlace Plane \
        -quality 88 \
        "$TMP_OUTPUT"

    test -s "$TMP_OUTPUT"
    DIMENSIONS=$(identify -format '%wx%h' "$TMP_OUTPUT")
    BYTES=$(stat -c '%s' "$TMP_OUTPUT")
    THUMBNAIL_SHA256=$(sha256sum "$TMP_OUTPUT" | awk '{print $1}')
    mv -f "$TMP_OUTPUT" "$OUTPUT"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${CODES[$INDEX]}" \
        "${ROLES[$INDEX]}" \
        "${SAMPLE_IDS[$INDEX]}" \
        "${SOURCES[$INDEX]}" \
        "${INPUTS[$INDEX]}" \
        "$ACTUAL_SHA256" \
        "assets/error_analysis/${OUTPUTS[$INDEX]}" \
        "$THUMBNAIL_SHA256" \
        "$DIMENSIONS" \
        "$BYTES" >> "$TMP_MANIFEST"
done

test "$(wc -l < "$TMP_MANIFEST")" -eq 8
mv -f "$TMP_MANIFEST" "$MANIFEST"
trap - EXIT

file "$OUTPUT_ROOT"/*.jpg
sha256sum "$OUTPUT_ROOT"/*.jpg "$MANIFEST"
