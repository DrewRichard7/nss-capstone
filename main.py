# import libraries
from typing import Optional

import requests
from bs4 import BeautifulSoup


def get_status(endpoint: str) -> Optional[BeautifulSoup]:
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
    """
    Main function to run the script.
    """
    endpoint = "https://www.pgatour.com/stats"
    soup = get_status(endpoint)
    if soup is not None and not soup.empty:
        print("\nSoup found!")
    else:
        print("\nNO SOUP FOR YOU!")


if __name__ == "__main__":
    main()
