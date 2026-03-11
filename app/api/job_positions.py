from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..auth import get_current_user, require_admin
from ..database import get_db
from ..models import JobPosition, User
from ..schemas import (
    JobPositionCreate,
    JobPositionResponse,
    JobPositionsListResponse,
    JobPositionUpdate,
    TokenPayload,
)

router = APIRouter(prefix="/api/job-positions", tags=["job-positions"])


def _to_response(pos: JobPosition) -> JobPositionResponse:
    return JobPositionResponse(
        id=pos.id,
        name=pos.name,
        level=pos.level,
        description=pos.description,
    )


@router.get("", response_model=JobPositionsListResponse)
async def list_job_positions(
    db: Session = Depends(get_db),
    current_user: TokenPayload = Depends(get_current_user),
):
    positions = db.query(JobPosition).order_by(JobPosition.level.asc()).all()
    return JobPositionsListResponse(items=[_to_response(p) for p in positions], total=len(positions))


@router.post("", response_model=JobPositionResponse, status_code=status.HTTP_201_CREATED)
async def create_job_position(
    payload: JobPositionCreate,
    db: Session = Depends(get_db),
    current_admin: TokenPayload = Depends(require_admin),
):
    existing = db.query(JobPosition).filter(JobPosition.name == payload.name).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Job position name already exists")
    pos = JobPosition(name=payload.name, level=payload.level, description=payload.description)
    db.add(pos)
    db.commit()
    db.refresh(pos)
    return _to_response(pos)


@router.get("/{position_id}", response_model=JobPositionResponse)
async def get_job_position(
    position_id: int,
    db: Session = Depends(get_db),
    current_user: TokenPayload = Depends(get_current_user),
):
    pos = db.query(JobPosition).filter(JobPosition.id == position_id).first()
    if not pos:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job position not found")
    return _to_response(pos)


@router.patch("/{position_id}", response_model=JobPositionResponse)
async def update_job_position(
    position_id: int,
    payload: JobPositionUpdate,
    db: Session = Depends(get_db),
    current_admin: TokenPayload = Depends(require_admin),
):
    pos = db.query(JobPosition).filter(JobPosition.id == position_id).first()
    if not pos:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job position not found")
    if payload.name is not None and payload.name != pos.name:
        conflict = db.query(JobPosition).filter(JobPosition.name == payload.name).first()
        if conflict:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Job position name already exists")
        pos.name = payload.name
    if payload.level is not None:
        pos.level = payload.level
    if payload.description is not None:
        pos.description = payload.description
    db.commit()
    db.refresh(pos)
    return _to_response(pos)


@router.delete("/{position_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job_position(
    position_id: int,
    db: Session = Depends(get_db),
    current_admin: TokenPayload = Depends(require_admin),
):
    pos = db.query(JobPosition).filter(JobPosition.id == position_id).first()
    if not pos:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job position not found")
    # Nullify references before deleting (SQLite may not handle ON DELETE SET NULL via FK)
    db.query(User).filter(User.position_id == position_id).update({"position_id": None})
    db.delete(pos)
    db.commit()
