import pytest

import backend.database as database


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Run each test against a throwaway sqlite file instead of the real sifter.db."""
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "test_sifter.db"))
    database.init_db()
    yield


def test_save_portfolio_creates_and_returns_id():
    pid = database.save_portfolio("Retirement", [
        {"ticker": "AAPL", "shares": 10, "cost_basis": 150.0, "purchase_date": "2024-01-01"},
        {"ticker": "MSFT", "shares": 5, "cost_basis": None, "purchase_date": None},
    ])
    assert isinstance(pid, int)

    portfolio = database.get_portfolio(pid)
    assert portfolio["name"] == "Retirement"
    assert len(portfolio["holdings"]) == 2
    tickers = {h["ticker"] for h in portfolio["holdings"]}
    assert tickers == {"AAPL", "MSFT"}


def test_save_portfolio_update_replaces_holdings():
    pid = database.save_portfolio("Taxable", [
        {"ticker": "AAPL", "shares": 10, "cost_basis": 150.0, "purchase_date": None},
    ])

    same_id = database.save_portfolio("Taxable", [
        {"ticker": "GOOGL", "shares": 3, "cost_basis": 120.0, "purchase_date": None},
    ], portfolio_id=pid)

    assert same_id == pid
    portfolio = database.get_portfolio(pid)
    assert len(portfolio["holdings"]) == 1
    assert portfolio["holdings"][0]["ticker"] == "GOOGL"


def test_save_portfolio_allows_multiple_lots_same_ticker():
    pid = database.save_portfolio("DCA", [
        {"ticker": "AAPL", "shares": 5, "cost_basis": 100.0, "purchase_date": "2023-01-01"},
        {"ticker": "AAPL", "shares": 5, "cost_basis": 200.0, "purchase_date": "2024-01-01"},
    ])
    portfolio = database.get_portfolio(pid)
    assert len(portfolio["holdings"]) == 2
    assert all(h["ticker"] == "AAPL" for h in portfolio["holdings"])


def test_list_portfolios_includes_holding_count():
    database.save_portfolio("Retirement", [
        {"ticker": "AAPL", "shares": 10, "cost_basis": 150.0, "purchase_date": None},
        {"ticker": "MSFT", "shares": 5, "cost_basis": None, "purchase_date": None},
    ])
    database.save_portfolio("Empty Portfolio", [])

    portfolios = database.list_portfolios()
    by_name = {p["name"]: p for p in portfolios}
    assert by_name["Retirement"]["holding_count"] == 2
    assert by_name["Empty Portfolio"]["holding_count"] == 0


def test_get_portfolio_missing_returns_none():
    assert database.get_portfolio(99999) is None


def test_delete_portfolio_removes_it_and_its_holdings():
    pid = database.save_portfolio("Temp", [
        {"ticker": "AAPL", "shares": 1, "cost_basis": None, "purchase_date": None},
    ])
    database.delete_portfolio(pid)
    assert database.get_portfolio(pid) is None
