# Per-program code runner for coding-RL rewards (scripts/safe_grader.py). Programs run with no network,
# a read-only root, a tmpfs work dir and an unprivileged user; this image only needs Python + numpy
# (the cookbook harness imports numpy, and some reference solutions use it).
FROM python@sha256:2259debdc6af85626058a15a7bf44ad6a68b8c98c9ae82a02cf5cdb92f096d47
RUN pip install --no-cache-dir numpy==2.3.4
USER 65534:65534
