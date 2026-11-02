---
title: TRUSTA enterprise SSDs
summary: ADATA TRUSTA enterprise SSD product lines, interfaces and form factors, use cases, and their role in AI Scaler Toolkit.
updatedAt: 2026-10-29
---

# TRUSTA enterprise SSDs

## Their role in AST

AST treats the NVMe SSD as **part of the memory hierarchy**, not only a place to keep files:

| Use | Notes | See |
|---|---|---|
| NVMe tier of the KV cache | KV for long conversations and shared documents lives on the SSD and is not recomputed on revisit | [vLLM KV offload measurements](../benchmarks/vllm-kv-offload.md) |
| NVMe offload in training | DeepSpeed ZeRO-3 puts parameters or optimizer state on the SSD | [DeepSpeed ZeRO-3 offload](../training/deepspeed-offload.md) |
| Reading model weights | MoE inference pulls in experts scattered by routing; slot files when handing off the GPU | [Gen5 NVMe measurements](../benchmarks/gen5-ssd.md) |

The measurements in this wiki use the **T7P5** (PCIe Gen5).

## Product lines

| Model | Interface | Form factor |
|---|---|---|
| **T7P5** | PCIe Gen5 NVMe | U.2, E1.S, E3.S |
| **TD7P40** | PCIe Gen4 NVMe | U.2 |
| **TB3P40** | PCIe Gen4 NVMe | M.2 |
| **TD5S30** | SATA III | 2.5-inch |
| **TB3S30** | SATA III | M.2 |

The T7P5 is the flagship series, for high-throughput, low-latency AI and data-centre workloads.

## Features

- **NVMe** (PCIe Gen4/Gen5 models): lower latency, higher IOPS, better parallelism
- Optional **FDP** (Flexible Data Placement) and **SR-IOV** (Single Root I/O Virtualization)
- Reliability: enterprise-grade components, error correction, power-loss protection, thermal management, wear levelling

## Use cases

| Use case | For |
|---|---|
| AI and data science | AI training and inference, vector databases, HPC |
| Business servers and virtualization | VMs, containers, VDI, web platforms |
| Object storage | AI, HPC, databases, backup, cloud |
| Databases and real-time transactions | OLTP, OLAP, financial transactions |
| Data-centre boot drives | Fast server boot and stable OS operation |

## Related

- [Overview](../overview.md)
- [Measured results](../benchmarks/index.md)
