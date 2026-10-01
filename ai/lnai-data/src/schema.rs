use serde::{Deserialize, Serialize};

pub const DATASET_SCHEMA_VERSION: &str = "1.0.0";

pub const DEFAULT_COORDINATE_EPOCH: f64 = 2000.0;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ColumnSchema {
    pub name: &'static str,
    pub data_type: ColumnType,
    pub nullable: bool,
    pub units: Option<&'static str>,
    pub description: &'static str,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ColumnType {
    String,
    Float32,
    Float64,
    UInt32,
    UInt64,
    Boolean,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum SchemaView {
    Base,
    Pinn,
    GnnKinematics,
    GnnLocalization,
    Siren,
}

pub fn canonical_columns() -> Vec<ColumnSchema> {
    vec![
        ColumnSchema {
            name: "source_id",
            data_type: ColumnType::String,
            nullable: false,
            units: None,
            description: "Unique stable source identifier (e.g., Gaia DR3 source_id)",
        },
        ColumnSchema {
            name: "ra_deg",
            data_type: ColumnType::Float64,
            nullable: false,
            units: Some("degrees"),
            description: "Right ascension (ICRS)",
        },
        ColumnSchema {
            name: "dec_deg",
            data_type: ColumnType::Float64,
            nullable: false,
            units: Some("degrees"),
            description: "Declination (ICRS)",
        },
        ColumnSchema {
            name: "epoch_year",
            data_type: ColumnType::Float64,
            nullable: false,
            units: Some("year"),
            description: "Coordinate epoch (default 2000.0)",
        },
        ColumnSchema {
            name: "parallax_mas",
            data_type: ColumnType::Float64,
            nullable: true,
            units: Some("milliarcseconds"),
            description: "Parallax",
        },
        ColumnSchema {
            name: "pm_ra_mas_yr",
            data_type: ColumnType::Float64,
            nullable: true,
            units: Some("mas/year"),
            description: "Proper motion in right ascension",
        },
        ColumnSchema {
            name: "pm_dec_mas_yr",
            data_type: ColumnType::Float64,
            nullable: true,
            units: Some("mas/year"),
            description: "Proper motion in declination",
        },
        ColumnSchema {
            name: "radial_velocity_kms",
            data_type: ColumnType::Float64,
            nullable: true,
            units: Some("km/s"),
            description: "Radial velocity",
        },
        ColumnSchema {
            name: "vx_kms",
            data_type: ColumnType::Float32,
            nullable: true,
            units: Some("km/s"),
            description: "Cartesian velocity X (Galactic)",
        },
        ColumnSchema {
            name: "vy_kms",
            data_type: ColumnType::Float32,
            nullable: true,
            units: Some("km/s"),
            description: "Cartesian velocity Y (Galactic)",
        },
        ColumnSchema {
            name: "vz_kms",
            data_type: ColumnType::Float32,
            nullable: true,
            units: Some("km/s"),
            description: "Cartesian velocity Z (Galactic)",
        },
        ColumnSchema {
            name: "mag_g",
            data_type: ColumnType::Float32,
            nullable: true,
            units: Some("mag"),
            description: "Stellar magnitude in G-band",
        },
        ColumnSchema {
            name: "mag_bp",
            data_type: ColumnType::Float32,
            nullable: true,
            units: Some("mag"),
            description: "Stellar magnitude in BP-band",
        },
        ColumnSchema {
            name: "mag_rp",
            data_type: ColumnType::Float32,
            nullable: true,
            units: Some("mag"),
            description: "Stellar magnitude in RP-band",
        },
        ColumnSchema {
            name: "ruwe",
            data_type: ColumnType::Float32,
            nullable: true,
            units: None,
            description: "RUWE (Renormalised Unit Weight Error)",
        },
        ColumnSchema {
            name: "astrometric_excess_noise",
            data_type: ColumnType::Float32,
            nullable: true,
            units: None,
            description: "Astrometric excess noise",
        },
        ColumnSchema {
            name: "is_valid",
            data_type: ColumnType::Boolean,
            nullable: false,
            units: None,
            description: "Boolean flag: whether the object passed basic quality filtering",
        },
        ColumnSchema {
            name: "neighbor_source_id",
            data_type: ColumnType::String,
            nullable: true,
            units: None,
            description: "[View-Only] Neighboring object ID",
        },
        ColumnSchema {
            name: "rel_x",
            data_type: ColumnType::Float32,
            nullable: true,
            units: Some("pc"),
            description: "[View-Only] Relative X coordinate from anchor",
        },
        ColumnSchema {
            name: "rel_y",
            data_type: ColumnType::Float32,
            nullable: true,
            units: Some("pc"),
            description: "[View-Only] Relative Y coordinate from anchor",
        },
        ColumnSchema {
            name: "rel_z",
            data_type: ColumnType::Float32,
            nullable: true,
            units: Some("pc"),
            description: "[View-Only] Relative Z coordinate from anchor",
        },
        ColumnSchema {
            name: "is_visible",
            data_type: ColumnType::Boolean,
            nullable: true,
            units: None,
            description: "[View-Only] Masking flag (visible/hidden) for localization",
        },
    ]
}

pub fn required_columns_for_view(view: &SchemaView) -> Vec<&'static str> {
    match view {
        SchemaView::Base => vec!["source_id", "ra_deg", "dec_deg", "epoch_year", "is_valid"],

        SchemaView::Pinn => vec![
            "source_id",
            "ra_deg",
            "dec_deg",
            "parallax_mas",
            "vx_kms",
            "vy_kms",
            "vz_kms",
            "is_valid",
        ],

        SchemaView::GnnKinematics => vec![
            "source_id",
            "ra_deg",
            "dec_deg",
            "parallax_mas",
            "pm_ra_mas_yr",
            "pm_dec_mas_yr",
            "radial_velocity_kms",
            "is_valid",
        ],

        SchemaView::GnnLocalization => vec![
            "source_id",
            "ra_deg",
            "dec_deg",
            "parallax_mas",
            "mag_g",
            "mag_bp",
            "mag_rp",
            "ruwe",
            "is_valid",
            "neighbor_source_id",
            "rel_x",
            "rel_y",
            "rel_z",
            "is_visible",
        ],

        SchemaView::Siren => vec!["source_id", "ra_deg", "dec_deg", "mag_g", "is_valid"],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_schema_is_golden_v1() {
        let columns = canonical_columns();
        let rendered: Vec<String> = columns
            .iter()
            .map(|c| {
                format!(
                    "{}:{:?}:nullable={}:units={}",
                    c.name,
                    c.data_type,
                    c.nullable,
                    c.units.unwrap_or("-")
                )
            })
            .collect();
        assert_eq!(
            rendered,
            vec![
                "source_id:String:nullable=false:units=-",
                "ra_deg:Float64:nullable=false:units=degrees",
                "dec_deg:Float64:nullable=false:units=degrees",
                "epoch_year:Float64:nullable=false:units=year",
                "parallax_mas:Float64:nullable=true:units=milliarcseconds",
                "pm_ra_mas_yr:Float64:nullable=true:units=mas/year",
                "pm_dec_mas_yr:Float64:nullable=true:units=mas/year",
                "radial_velocity_kms:Float64:nullable=true:units=km/s",
                "vx_kms:Float32:nullable=true:units=km/s",
                "vy_kms:Float32:nullable=true:units=km/s",
                "vz_kms:Float32:nullable=true:units=km/s",
                "mag_g:Float32:nullable=true:units=mag",
                "mag_bp:Float32:nullable=true:units=mag",
                "mag_rp:Float32:nullable=true:units=mag",
                "ruwe:Float32:nullable=true:units=-",
                "astrometric_excess_noise:Float32:nullable=true:units=-",
                "is_valid:Boolean:nullable=false:units=-",
                "neighbor_source_id:String:nullable=true:units=-",
                "rel_x:Float32:nullable=true:units=pc",
                "rel_y:Float32:nullable=true:units=pc",
                "rel_z:Float32:nullable=true:units=pc",
                "is_visible:Boolean:nullable=true:units=-",
            ]
        );
    }

    #[test]
    fn required_columns_reference_canonical_names() {
        let canonical: Vec<&str> = canonical_columns().iter().map(|c| c.name).collect();
        for view in [
            SchemaView::Base,
            SchemaView::Pinn,
            SchemaView::GnnKinematics,
            SchemaView::GnnLocalization,
            SchemaView::Siren,
        ] {
            for name in required_columns_for_view(&view) {
                assert!(
                    canonical.contains(&name),
                    "view {view:?} requires unknown column '{name}'"
                );
            }
        }
    }

    #[test]
    fn schema_types_round_trip_through_json() {
        for column in canonical_columns() {
            let first = serde_json::to_value(&column).expect("serialize column");
            let second = serde_json::to_value(&column).expect("re-serialize column");
            assert_eq!(first, second, "serialization must be deterministic");
            let mut keys: Vec<&str> = first
                .as_object()
                .expect("object")
                .keys()
                .map(String::as_str)
                .collect();
            keys.sort_unstable();
            assert_eq!(
                keys,
                vec!["data_type", "description", "name", "nullable", "units"]
            );
        }
        for view in [
            SchemaView::Base,
            SchemaView::Pinn,
            SchemaView::GnnKinematics,
            SchemaView::GnnLocalization,
            SchemaView::Siren,
        ] {
            let json = serde_json::to_string(&view).expect("serialize view");
            let back: SchemaView = serde_json::from_str(&json).expect("deserialize view");
            assert_eq!(back, view);
        }
    }

    #[test]
    fn schema_version_and_epoch_are_frozen() {
        assert_eq!(DATASET_SCHEMA_VERSION, "1.0.0");
        assert_eq!(DEFAULT_COORDINATE_EPOCH, 2000.0);
    }
}
