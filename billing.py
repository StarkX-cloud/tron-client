billing_ledger = {}

def charge(user_id, amount):

    if user_id not in billing_ledger:
        billing_ledger[user_id] = 0

    billing_ledger[user_id] += amount


def get_balance(user_id):

    return billing_ledger.get(user_id, 0)