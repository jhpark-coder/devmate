"""
도메인 예외 정의

비즈니스 로직 위반 시 발생하는 예외들을 정의합니다.
"""


class DomainException(Exception):
    """도메인 계층 기본 예외"""
    pass


class DocumentNotFoundError(DomainException):
    """문서를 찾을 수 없을 때"""
    pass


class DocumentNotIndexedError(DomainException):
    """문서가 아직 인덱싱되지 않았을 때"""
    pass


class InvalidDocumentStatusError(DomainException):
    """잘못된 문서 상태 전환 시도"""
    pass


class ConversationNotFoundError(DomainException):
    """대화를 찾을 수 없을 때"""
    pass


class InvalidMessageRoleError(DomainException):
    """잘못된 메시지 역할"""
    pass


class EmptyDocumentError(DomainException):
    """빈 문서 (청크가 없음)"""
    pass
