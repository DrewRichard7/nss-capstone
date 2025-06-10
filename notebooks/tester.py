import requests
from typing import Optional
from bs4 import BeautifulSoup


def fetch_data(endpoint: str) -> Optional[BeautifulSoup]:
    """
    Get the status of an endpoint.
    Args:
        endpoint (str): the url of the endpoint
    Returns:
        str: the status of the endpoint
    """
    try:
        response = requests.get(endpoint)
        status = response.status_code
        if status == 200:
            print(f"\nResponse code: {status}\nStatus OK")
            soup = BeautifulSoup(response.text, features="html.parser")
            return soup
        else:
            print(f"Response code: {status}. Status not OK")
            return None
    except Exception as e:
        print(f"Error getting endpoint: {e}")
        return None


def main():
    url = "https://www.mlb.com/stats/"
    soup = fetch_data(url)
    if soup is not None and not soup.empty:
        print("there's soup in this soup")
    else:
        print("no soup for you")


if __name__ == "__main__":
    main()
