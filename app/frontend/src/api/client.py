import json
import os

import requests


class APIClient:
    _instance = None

    def __new__(cls, base_url="http://0.0.0.0:6001"):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.base_url = base_url
        return cls._instance

    def get(self, target: str, endpoint: str, params=None):
        """GET 요청을 보내는 메서드

        Args:
            target (str): 요청을 보낼 대상 (e.g. hub, task, dataset, etc.)
            endpoint (str): 요청을 보낼 엔드포인트 (e.g. /train, /evaluate, /inference, etc.)
            params (dict): 요청에 포함할 쿼리 스트링
        """
        try:
            response = requests.get(f"{self.base_url}/{target}/{endpoint}", params=params)
            response.raise_for_status()  # HTTP 요청 에러를 확인
            return response.json()  # JSON 응답을 파싱하여 반환
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")  # HTTP 에러 발생 시
        except Exception as err:
            print(f"An error occurred: {err}")  # 기타 에러

    def post(self, target: str, endpoint: str, params=None):
        """POST 요청을 보내는 메서드

        Args:
            target (str): 요청을 보낼 대상 (e.g. hub, task, dataset, etc.)
            endpoint (str): 요청을 보낼 엔드포인트 (e.g. /train, /evaluate, /inference, etc.)
            params (dict): 요청에 포함할 쿼리 스트링
        """
        try:
            response = requests.post(f"{self.base_url}/{target}/{endpoint}", json=params)
            response.raise_for_status()  # HTTP 요청 에러를 확인
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")  # HTTP 에러 발생 시
        except Exception as err:
            print(f"An error occurred: {err}")  # 기타 에러
