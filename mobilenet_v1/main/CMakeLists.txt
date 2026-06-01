idf_component_register(
    SRCS 
        "mobilenet_v1.cpp"
        "models/mobilenet_v1_025_int8.cc"                   
        "models/mobilenet_v1_050_int8.cc"
        "models/mobilenet_v1_075_int8.cc"
        "models/mobilenet_v1_100_int8.cc"
    PRIV_REQUIRES 
        spi_flash
        esp_psram
        esp_timer
    INCLUDE_DIRS 
        "."
        "models"
)

