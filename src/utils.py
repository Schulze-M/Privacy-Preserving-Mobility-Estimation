def validate_coordinates(latitude: float, longitude: float) -> bool:
    '''
    Validate geographic coordinates.
    
    :param latitude: Latitude value (must be between -90 and 90).
    :param longitude: Longitude value (must be between -180 and 180).
    :return: True if valid, False otherwise.
    '''

    if -90 <= latitude <= 90 and -180 <= longitude <= 180:
        return True
    return False