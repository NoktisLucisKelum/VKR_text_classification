import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import re


def standardize_chemical_formula(text):
    """
    Standardizes chemical formulas in text using RDKit.
    Formulas will be enclosed in [CHEM]...[/CHEM] tags.
    """

    def replace_formula(match):
        formula = match.group(0)
        try:
            mol = Chem.MolFromSmiles(formula)
            if mol:
                # Generate 2D coordinates
                mol = Chem.AddHs(mol)  # Add hydrogens for better depiction
                rdMolDraw2D.PrepareMolForDrawing(mol)

                formula = Chem.MolToSmiles(mol, isomericSmiles=False)
                return f"[CHEM]{formula}[/CHEM]"
        except:
            pass
        return f"[CHEM]{formula}[/CHEM]"

    return re.sub(r'([A-Z][a-z]?\d*(\(|\[|\{).*?(\)|\]|\})|[A-Z][a-z]?\d*[^A-Za-z\s]+)', replace_formula, text)


# Example usage
df = pd.DataFrame({
    'text': ['Ni[75]Al[15]Ti[10,',
             ' 4,6-диарил-1,6-дигидропиридин-2(3Н)-онов',
             'pH 5,5, Na{+}/Mg{2+}',
             'один два три',
             'Mil53(Al), Na{+}, Na{+} и Mg{2+} 22,0 и 0,6 моль~xсм{-2}~xс{-1}, раз два раз два',
             '(Mo, Ni, Cr, Al, Sn, B), фенил-2,4,6-триметилбензоилфосфината натрия, erqghwteyr']
})

df['standardized_text'] = df['text'].apply(standardize_chemical_formula)
print(df)
