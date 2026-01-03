#!/usr/bin/env python
"""
Test to verify price difference calculation makes sense
"""
import numpy as np

# Example scenario
base_price = 10.0
own_scalar = 0.8  # Actor output
comp_scalar = 0.6  # Competitor actor output

# How prices are actually set in environment
own_price = 2 * base_price * own_scalar  # = 2 * 10 * 0.8 = 16.0
comp_price = 2 * base_price * comp_scalar  # = 2 * 10 * 0.6 = 12.0

print("="*60)
print("PRICE CALCULATION VERIFICATION")
print("="*60)
print(f"\nBase price: ${base_price:.2f}")
print(f"Own scalar (actor output): {own_scalar}")
print(f"Competitor scalar: {comp_scalar}")
print(f"\nActual prices:")
print(f"  Own price: 2 × {base_price} × {own_scalar} = ${own_price:.2f}")
print(f"  Competitor price: 2 × {base_price} × {comp_scalar} = ${comp_price:.2f}")

# Current feature calculation
price_diff_current = (own_price - comp_price) / base_price
print(f"\n" + "="*60)
print(f"CURRENT IMPLEMENTATION:")
print(f"="*60)
print(f"price_difference = (own_price - comp_price) / base_price")
print(f"                 = ({own_price} - {comp_price}) / {base_price}")
print(f"                 = {price_diff_current:.2f}")
print(f"\nThis equals: 2 × (own_scalar - comp_scalar)")
print(f"           = 2 × ({own_scalar} - {comp_scalar})")
print(f"           = 2 × {own_scalar - comp_scalar:.2f}")
print(f"           = {2 * (own_scalar - comp_scalar):.2f} ✓")

# Alternative: ratio
price_ratio = own_price / comp_price
print(f"\n" + "="*60)
print(f"ALTERNATIVE: PRICE RATIO")
print(f"="*60)
print(f"price_ratio = own_price / comp_price")
print(f"            = {own_price} / {comp_price}")
print(f"            = {price_ratio:.3f}")
print(f"\nInterpretation: Own price is {(price_ratio-1)*100:.1f}% {'higher' if price_ratio > 1 else 'lower'} than competitor")

# Which makes more sense?
print(f"\n" + "="*60)
print(f"INTERPRETATION")
print(f"="*60)
print(f"\nCurrent (normalized difference = {price_diff_current:.2f}):")
print(f"  ✓ Tells model: 'You're charging ${price_diff_current * base_price:.2f} more than competitor'")
print(f"  ✓ Scale-invariant across different OD pairs")
print(f"  ✓ Symmetric: +0.4 means charging more, -0.4 means charging less")
print(f"  ✓ Zero-centered: 0 means equal pricing")
print(f"\nRatio (= {price_ratio:.3f}):")
print(f"  ✓ Tells model: 'You're charging {price_ratio:.2f}x the competitor price'")
print(f"  ✓ Asymmetric: ratio=2.0 (2x) vs ratio=0.5 (0.5x) not equally distant from 1.0")
print(f"  ✗ Not zero-centered: 1.0 means equal pricing")
print(f"  ✗ Less intuitive for learning")

print(f"\n" + "="*60)
print(f"RECOMMENDATION")
print(f"="*60)
print(f"✓ Current implementation (normalized difference) is CORRECT")
print(f"✓ The '2×' factor is consistent with how environment sets prices")
print(f"✓ Feature correctly represents competitive position")
print(f"✗ NO CHANGES NEEDED")
print(f"="*60)
