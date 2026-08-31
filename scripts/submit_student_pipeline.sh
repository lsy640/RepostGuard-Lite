#!/bin/bash

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/msai/xjiang026/projects/repostguard-lite}
cd "$PROJECT_ROOT"

mkdir -p logs outputs/community_forensics/student_mnv3_dual_teacher
test -x /home/msai/xjiang026/.conda/envs/repostguard/bin/python
test -s data/raw/community_forensics/COMPLETE
test -s data/manifests/community_forensics_train.csv

DATA_JOB=$(sbatch --parsable scripts/slurm/validate_student_data.sbatch)
PREFETCH_JOB=$(sbatch --parsable scripts/slurm/prefetch_distillation_backbones.sbatch)
TEACHER_JOB=$(sbatch --parsable \
    --dependency="afterok:${DATA_JOB}:${PREFETCH_JOB}" \
    scripts/slurm/train_distillation_teachers.sbatch)
TEACHER_GATE_JOB=$(sbatch --parsable \
    --dependency="afterok:${TEACHER_JOB}" \
    scripts/slurm/evaluate_distillation_teachers.sbatch)
TEACHER_CACHE_JOB=$(sbatch --parsable \
    --dependency="afterok:${TEACHER_GATE_JOB}" \
    scripts/slurm/cache_student_teachers.sbatch)
CALIBRATION_JOB=$(sbatch --parsable \
    --dependency="afterok:${TEACHER_GATE_JOB}" \
    scripts/slurm/cache_and_calibrate_student_teachers.sbatch)
SMOKE_JOB=$(sbatch --parsable \
    --dependency="afterok:${TEACHER_CACHE_JOB}:${CALIBRATION_JOB}" \
    scripts/slurm/smoke_student_distill.sbatch)
STUDENT_JOB=$(sbatch --parsable \
    --dependency="afterok:${SMOKE_JOB}" \
    scripts/slurm/train_student_distill.sbatch)
EVALUATION_JOB=$(sbatch --parsable \
    --dependency="afterok:${STUDENT_JOB}" \
    scripts/slurm/evaluate_student_distill.sbatch)

{
    printf 'data_validation=%s\n' "$DATA_JOB"
    printf 'backbone_prefetch=%s\n' "$PREFETCH_JOB"
    printf 'teacher_training=%s\n' "$TEACHER_JOB"
    printf 'teacher_acceptance=%s\n' "$TEACHER_GATE_JOB"
    printf 'teacher_train_cache=%s\n' "$TEACHER_CACHE_JOB"
    printf 'teacher_calibration=%s\n' "$CALIBRATION_JOB"
    printf 'student_smoke=%s\n' "$SMOKE_JOB"
    printf 'student_training=%s\n' "$STUDENT_JOB"
    printf 'student_evaluation=%s\n' "$EVALUATION_JOB"
} | tee outputs/community_forensics/student_mnv3_dual_teacher/pipeline_jobs.txt
