from __future__ import annotations

from pathlib import Path
from typing import Iterable
from datetime import datetime

import pandas as pd
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    create_engine,
    select,
)
from sqlalchemy.orm import declarative_base, Session

Base = declarative_base()


class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True)
    uri_or_path = Column(String, unique=True)
    label = Column(String)
    split = Column(String)
    verified = Column(Boolean, default=False)
    notes = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer)
    label = Column(String)
    prob = Column(Float)
    gradcam = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


def get_engine(dsn: str):
    return create_engine(dsn, future=True)


def init_db(dsn: str) -> None:
    engine = get_engine(dsn)
    Base.metadata.create_all(engine)


def sync_from_folder_or_csv(dsn: str, root: str | None = None, csv: str | None = None) -> None:
    engine = get_engine(dsn)
    Base.metadata.create_all(engine)
    with Session(engine) as sess:
        if root:
            for cls_dir in Path(root).glob("*"):
                if not cls_dir.is_dir():
                    continue
                label = cls_dir.name
                for img in cls_dir.glob("*.jpg"):
                    if not sess.execute(select(Image).filter_by(uri_or_path=str(img))).scalar_one_or_none():
                        sess.add(Image(uri_or_path=str(img), label=label, split="train"))
        if csv:
            df = pd.read_csv(csv)
            for _, row in df.iterrows():
                if not sess.execute(select(Image).filter_by(uri_or_path=row["path"])).scalar_one_or_none():
                    sess.add(Image(uri_or_path=row["path"], label=row["label"], split=row.get("split", "train")))
        sess.commit()


def push_predictions(dsn: str, records: Iterable[dict]) -> None:
    engine = get_engine(dsn)
    with Session(engine) as sess:
        for rec in records:
            sess.add(Prediction(**rec))
        sess.commit()


def pull_verified_labels(dsn: str) -> pd.DataFrame:
    engine = get_engine(dsn)
    with Session(engine) as sess:
        rows = sess.execute(select(Image).filter_by(verified=True)).scalars().all()
    return pd.DataFrame([
        {"path": r.uri_or_path, "label": r.label, "split": r.split} for r in rows
    ])
