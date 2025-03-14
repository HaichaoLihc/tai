from typing import List

from pydantic import BaseModel


class Course(BaseModel):
    courseId: str
    courseName: str
    isPublic: bool


class Meta(BaseModel):
    page: int
    limit: int
    total: int
    totalPages: int


class CoursesResponse(BaseModel):
    data: List[Course]
    meta: Meta
