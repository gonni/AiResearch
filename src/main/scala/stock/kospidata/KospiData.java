package stock.kospidata;

import lombok.AllArgsConstructor;
import lombok.Data;

@AllArgsConstructor
@Data
public class KospiData {
    private String date ;
    private double indexValue ;
    private double totalEa ;
    private double totalVolume ;
    private double ant ;
    private double foreigner ;
    private double company ;
    private double investBank ;
    private double investTrust ;
    private double pensionFund ;

}
