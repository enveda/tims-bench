import re

# Monoisotopic atomic masses
ATOMIC_MASSES = {
    'H': 1.008,
    'C': 12.000,
    'N': 14.007,
    'O': 15.999,
    'S': 32.06,
    'P': 30.974,
    'F': 18.998,
    'Cl': 35.45,
    'Br': 79.904,
    'I': 126.90447,
    'Na': 22.98977,
    'K': 39.0983,
    'B': 10.81,
}


def parse_formula(formula: str) -> float:
    """
    Parse a molecular formula and compute its monoisotopic mass.
    
    Args:
        formula: A molecular formula string (e.g., 'H2O', 'C6H12O6', 'NH4')
    
    Returns:
        The monoisotopic mass of the formula
    """
    if not formula:
        return 0.0
    
    # Pattern to match element symbols followed by optional counts
    # Handles: H, H2, Cl, Cl2, Na, etc.
    pattern = r'([A-Z][a-z]?)(\d*)'
    
    total_mass = 0.0
    for match in re.finditer(pattern, formula):
        element = match.group(1)
        count_str = match.group(2)
        count = int(count_str) if count_str else 1
        
        if element in ATOMIC_MASSES:
            total_mass += ATOMIC_MASSES[element] * count
        elif element:
            # Unknown element - could raise an error or return None
            pass
    
    return total_mass


def calculate_adduct_mz(M: float, adduct: str) -> float:
    """
    Calculate the observed m/z from a neutral mass M and an adduct string.
    
    Args:
        M: The neutral molecular mass
        adduct: The adduct string (e.g., '[M+H]+', '[2M+Na]+', '[M+H-H2O]+')
    
    Returns:
        The calculated m/z value, or None if the adduct cannot be parsed
    """
    if not adduct or adduct == '':
        return None
    
    # ISF (In-Source Fragment) - return observed m/z as-is
    if adduct == 'ISF':
        return M
    
    # Handle Cat- adducts (cation fragments)
    # These are [M+H]+ that have lost neutral fragments
    # e.g., [Cat-H2O]+ means M + H - H2O
    if adduct.startswith('[Cat'):
        # Extract content inside brackets
        bracket_match = re.search(r'\[(.*?)\]', adduct)
        if not bracket_match:
            return None
        
        content = bracket_match.group(1)
        
        # Start with M + H (protonated molecule)
        mass = M + ATOMIC_MASSES['H']
        
        # Remove "Cat" and process subtractions
        content = content.replace('Cat', '')
        
        # Find all subtractions (after -)
        subtractions = re.findall(r'-([^-]+)', content)
        
        for sub in subtractions:
            sub = sub.strip()
            # Check for multiplier
            mult_match = re.match(r'^(\d+)([A-Z].*)$', sub)
            if mult_match:
                multiplier = int(mult_match.group(1))
                group = mult_match.group(2)
            else:
                multiplier = 1
                group = sub
            
            mass -= multiplier * parse_formula(group)
        
        return mass
    
    # Extract charge
    charge = 1
    charge_match = re.search(r'\](\d*)([+-])(\d*)$', adduct)
    if charge_match:
        pre_sign = charge_match.group(1)
        sign = charge_match.group(2)
        post_sign = charge_match.group(3)
        
        if pre_sign:
            charge = int(pre_sign)
        elif post_sign:
            charge = int(post_sign)
        else:
            charge = 1
        
        if sign == '-':
            charge = -charge
    
    # Extract the content inside brackets
    bracket_match = re.search(r'\[(.*?)\]', adduct)
    if not bracket_match:
        return None
    
    content = bracket_match.group(1)
    
    # Determine M multiplier (e.g., 2M, 3M)
    m_multiplier = 1
    m_match = re.match(r'(\d*)M', content)
    if m_match and m_match.group(1):
        m_multiplier = int(m_match.group(1))
    
    # Start with base mass
    mass = m_multiplier * M
    
    # Remove the M part and process additions/subtractions
    content = re.sub(r'^\d*M', '', content)
    
    # Find all additions (after +) and subtractions (after -)
    additions = re.findall(r'\+([^+-]+)', content)
    subtractions = re.findall(r'-([^+-]+)', content)
    
    # Process additions
    for add in additions:
        add = add.strip()
        # Check for multiplier (e.g., 2H2O, 3H)
        mult_match = re.match(r'^(\d+)([A-Z].*)$', add)
        if mult_match:
            multiplier = int(mult_match.group(1))
            group = mult_match.group(2)
        else:
            multiplier = 1
            group = add
        
        mass += multiplier * parse_formula(group)
    
    # Process subtractions
    for sub in subtractions:
        sub = sub.strip()
        # Check for multiplier
        mult_match = re.match(r'^(\d+)([A-Z].*)$', sub)
        if mult_match:
            multiplier = int(mult_match.group(1))
            group = mult_match.group(2)
        else:
            multiplier = 1
            group = sub
        
        mass -= multiplier * parse_formula(group)
    
    # Divide by absolute charge for m/z
    mz = mass / abs(charge)
    
    return mz
