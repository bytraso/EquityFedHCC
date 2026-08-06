from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from equityfedhcc.data.contracts import DatasetDescriptor, ModalityIndex, PatientRecord

EICU = DatasetDescriptor(
    name="eICU-CRD",
    version="2.0",
    license_name="PhysioNet Credentialed Health Data License 1.5.0",
    source_url="https://physionet.org/content/eicu-crd/2.0/",
    restricted=True,
    modalities=(ModalityIndex.EHR, ModalityIndex.BIOMARKERS),
    partition_field="hospitalid",
)

TCGA_LIHC = DatasetDescriptor(
    name="TCGA-LIHC",
    version="GDC harmonized 36+",
    license_name="TCGA Open Access Tier",
    source_url="https://portal.gdc.cancer.gov/projects/TCGA-LIHC",
    restricted=False,
    modalities=(
        ModalityIndex.EHR,
        ModalityIndex.CT,
        ModalityIndex.PATHOLOGY,
        ModalityIndex.OMICS,
        ModalityIndex.BIOMARKERS,
    ),
    partition_field="tissue_source_site",
)

NHANES = DatasetDescriptor(
    name="NHANES Continuous",
    version="2013-2023",
    license_name="United States Government Public Domain",
    source_url="https://www.cdc.gov/nchs/nhanes/",
    restricted=False,
    modalities=(ModalityIndex.EHR, ModalityIndex.BIOMARKERS),
    partition_field="cycle",
)

LITS_MSD = DatasetDescriptor(
    name="LiTS and MSD Task03 Liver",
    version="LiTS 2017 and MSD 2018",
    license_name="CC BY-NC-SA 4.0 and CC BY-SA 4.0",
    source_url="http://medicaldecathlon.com/",
    restricted=False,
    modalities=(ModalityIndex.CT,),
    partition_field="institution",
)


@dataclass(frozen=True)
class EICUColumns:
    patient_id: str = "patientunitstayid"
    hospital_id: str = "hospitalid"
    diagnosis_code: str = "diagnosisstring"
    age: str = "age"
    sex: str = "gender"
    race: str = "ethnicity"


def is_cirrhosis_code(code: str) -> bool:
    normalized = code.strip().upper().replace(".", "")
    if len(normalized) < 3 or not normalized.startswith("K"):
        return False
    try:
        family = int(normalized[1:3])
    except ValueError:
        return False
    return 70 <= family <= 77


def load_eicu_index(path: Path, columns: EICUColumns | None = None) -> pd.DataFrame:
    columns = columns or EICUColumns()
    frame = pd.read_csv(path)
    required = {
        columns.patient_id,
        columns.hospital_id,
        columns.diagnosis_code,
        columns.age,
        columns.sex,
        columns.race,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing eICU columns: {sorted(missing)}")
    cirrhosis = frame[columns.diagnosis_code].fillna("").map(is_cirrhosis_code)
    selected = frame.loc[cirrhosis].copy()
    selected = selected[selected[columns.age].astype(str) != ""]
    return selected


def nhanes_stratum(ridreth3: int) -> str:
    mapping = {
        1: "mexican_american",
        2: "other_hispanic",
        3: "non_hispanic_white",
        4: "non_hispanic_black",
        6: "non_hispanic_asian",
        7: "other_multiracial",
    }
    return mapping.get(ridreth3, "unknown")


def records_from_tabular(
    frame: pd.DataFrame,
    patient_column: str,
    site_column: str,
    stratum_column: str,
    label_column: str,
    feature_columns: Iterable[str],
) -> list[PatientRecord]:
    features = tuple(feature_columns)
    records: list[PatientRecord] = []
    for row in frame.itertuples(index=False):
        values = row._asdict()
        tensor = torch.tensor([float(values[name]) for name in features], dtype=torch.float32)
        records.append(
            PatientRecord(
                patient_id=str(values[patient_column]),
                site_id=str(values[site_column]),
                stratum_id=str(values[stratum_column]),
                label=float(values[label_column]),
                biomarkers=tensor,
            )
        )
    return records
