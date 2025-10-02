#Refractive Index Database (RIDB) Python API
import yaml, difflib
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from importlib.resources import files as pkg_files, path
from typing import Any, Dict, Optional

from .utils import convert_spectral, lorentzfunc

@dataclass
class Material:
    '''Class to handle material data from RIDB.'''
    path: Path                              # path to the YAML file
    
    # filled in after init
    metadata: Dict[str, Any] = field(init=False)
    rawdata: np.ndarray = field(init=False)

    def __post_init__(self):
        # Normalize/resolve the YAML path
        self.path = Path(self.path)
        if not self.path.suffix:
            # allow passing a stem; assume YAML
            self.path = self.path.with_suffix(".yaml")
        if not self.path.exists():
            raise FileNotFoundError(f"YAML not found: {self.path}")

        # Derive name from file stem if not provided
        self.name = self.path.stem

        # Load metadata and raw data
        self.metadata = self._get_metadata(self.path)
        self.rawdata = self._get_rawdata(self.path)

    @staticmethod
    def _get_metadata(yaml_path: Path):
        with yaml_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _raw_path_for(yaml_path: Path) -> Path:
        """Replace .yaml/.yml with .npy in the same folder."""
        if yaml_path.suffix.lower() in {".yaml", ".yml"}:
            return yaml_path.with_suffix(".npy")
        # fallback if someone passed a weird extension
        return yaml_path.parent / (yaml_path.stem + ".npy")

    def _get_rawdata(self, yaml_path: Path) -> np.ndarray:
        npy_path = self._raw_path_for(yaml_path)
        if not npy_path.exists():
            raise FileNotFoundError(
                f"Raw data .npy not found next to YAML:\n  YAML: {yaml_path}\n  NPY : {npy_path}"
            )
        return np.load(npy_path)  # add mmap_mode="r" if files are large
    
    def n(self, wvl: np.ndarray, units: str = 'nm'):
        '''
        Get complex refractive index n+ik at specified wavelengths.
        '''
        wvl_conv = self.convert_to(wvl, units)      # convert input wavelength to base units
        rev = wvl_conv[0] > wvl_conv[-1]           # remember if user asked for descending
        if rev:
            wvl_conv = wvl_conv[::-1]
            rawdata = self.rawdata[::-1]
        else:
            rawdata = self.rawdata

        n = np.interp(wvl_conv, rawdata[:,0], rawdata[:,1])
        k = np.interp(wvl_conv, rawdata[:,0], rawdata[:,2])
        
        if rev:                                    # restore original order
            n = n[::-1]; k = k[::-1]
            
        return n + 1j*k
    
    def epsilon(self, wvl: np.ndarray, units: str = 'nm'):
        '''
        Get complex permittivity (dielectric function) ε = (n+ik)^2 at specified wavelengths.
        '''
        n = self.n(wvl, units)
        return n**2
    
    def epsilon_lorentz(self, wvl: np.ndarray, units: str = 'nm'):
        '''
        Get complex permittivity (dielectric function) ε = (n+ik)^2 at specified wavelengths
        using Lorentzian fit parameters from metadata.
        '''
        if 'lorentz_params' not in self.metadata.get('data', {}):
            raise ValueError(f"Material '{self.name}' does not have Lorentzian fit parameters.")

        wvl_conv = convert_spectral(wvl, units, '1/um')      # convert input wavelength to base units
        rev = wvl_conv[0] > wvl_conv[-1]           # remember if user asked for descending
        if rev:
            wvl_conv = wvl_conv[::-1]

        parameters = []
        for name, value in self.metadata['data']['lorentz_params']['lorentzians'].items():
            parameters.append(np.array(list(value)).flatten())
            
        p = np.concatenate(parameters)
        eps_inf = self.metadata['data']['lorentz_params']['eps_inf']
        
        eps_fit = lorentzfunc(p, wvl_conv) + eps_inf  #calculate fitted epsilon
         
        if rev:                                    # restore original order
            eps_fit = eps_fit[::-1]
            
        return eps_fit
    
    def n_lorentz(self, wvl: np.ndarray, units: str = 'nm'):
        '''
        Get complex refractive index n+ik at specified wavelengths
        using Lorentzian fit parameters from metadata.
        '''
        eps = self.epsilon_lorentz(wvl, units)
        n = np.sqrt(eps)
        return n
    
    def convert_to(self, x, target_units: str):
        '''Convert input x (wavelength/frequency/energy) to the base units of the material data.'''
        cols = self.metadata['data']['columns']         # e.g. ['wl_nm','n','k']
        base_q = cols[0]                                 # first quantity (wavelength/freq/energy)
        base_units = self.metadata['data']['units'][base_q]  # e.g. 'nm'
        return convert_spectral(x, target_units, base_units)
    
    def to_meep(self):
        '''Convert Lorentzian fit parameters to a Meep Medium object.'''
        #Try to import meep only when this function is called
        try:
            import meep as mp
        except ImportError:
            raise ImportError("meep library is not installed. Please install it to use this feature.")
        
        # Check if Lorentzian parameters are available
        if 'lorentz_params' not in self.metadata.get('data', {}):
            raise ValueError(f"Material '{self.name}' does not have Lorentzian fit parameters.")
        
        parameters = []
        for name, value in self.metadata['data']['lorentz_params']['lorentzians'].items():
            parameters.append(np.array(list(value)).flatten())
            
        p = np.concatenate(parameters)
        eps_inf = self.metadata['data']['lorentz_params']['eps_inf']
        
        num_lorentzians = len(p) // 3
        
        # Define a `Medium` class object using the optimal fitting parameters.
        E_susceptibilities = []

        for n in range(num_lorentzians):
            mymaterial_freq = p[3 * n + 1]
            mymaterial_gamma = p[3 * n + 2]

            if mymaterial_freq == 0:
                mymaterial_sigma = p[3 * n + 0]
                E_susceptibilities.append(
                    mp.DrudeSusceptibility(
                        frequency=1.0, gamma=mymaterial_gamma, sigma=mymaterial_sigma
                    )
                )
            else:
                mymaterial_sigma = p[3 * n + 0] / mymaterial_freq**2
                E_susceptibilities.append(
                    mp.LorentzianSusceptibility(
                        frequency=mymaterial_freq,
                        gamma=mymaterial_gamma,
                        sigma=mymaterial_sigma,
                    )
                )

        mymaterial = mp.Medium(epsilon=eps_inf, E_susceptibilities=E_susceptibilities)
        
        return mymaterial
    
    def __repr__(self) -> str:
        return f"Material({self.name!r})"   # shown in REPL, lists, etc.

    def __str__(self) -> str:
        return self.name                    # shown by print(obj)
        
class RIDB:
    def __init__(self, folder: str | Path | None = None):
        if folder is None:
            # Resolve from the top-level package, not the subpackage:
            self.folder = pkg_files("ridb").joinpath("materials")
        else:
            self.folder = Path(folder)

        if not self.folder.exists():
            raise FileNotFoundError(f"RIDB materials folder not found: {self.folder}")
        
        self._recs = []
        self._by_name = {}
        self._by_file = {}
        self._build_index()

    def _build_index(self):
        self._recs.clear()
        for p in self.folder.glob("*.yaml"):
            try:
                meta = yaml.safe_load(p.read_text()) or {}
            except Exception:
                meta = {}
            rec = {
                "file": p.name,
                "path": str(p),
                "name": meta.get("name", "") or p.stem,
                "formula": meta.get("formula", "") or "",
                "source": meta.get("source", "") or "",
            }
            self._recs.append(rec)
        self._by_name = {r["name"]: r for r in self._recs}
        self._by_file = {r["file"]: r for r in self._recs}

    @property
    def materials(self):
        return sorted(r["name"] for r in self._recs)
    
    def get_material(self, name: str):
        if name in self.materials:
            return Material(self._by_name[name]["path"])
        else:
            raise ValueError(f"Material '{name}' not found in the database.")
        
    def find_materials(self, query: str, topk: int = 15):
        terms, filters = [], {}
        for tok in query.split():
            if ":" in tok:
                k, v = tok.split(":", 1)
                filters[k.lower()] = v.lower()
            else:
                terms.append(tok.lower())

        def score(rec):
            fields = [rec["name"], 
                      rec["formula"], 
                      rec["source"], 
                      rec["file"]]
            txt = " | ".join(fields).lower()

            # hard filters
            for k, v in filters.items():
                if k in rec and v not in str(rec[k]).lower():
                    return -1.0

            sub = sum(t in txt for t in terms)
            fuzz = max(difflib.SequenceMatcher(None, " ".join(terms), f.lower()).ratio()
                       for f in fields) if terms else 0.0
            return sub*2 + fuzz

        scored = [(score(r), r) for r in self._recs]
        scored = [x for x in scored if x[0] >= 0]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r["name"] for _, r in scored[:topk]]
    
    