import 'package:turd/utils/theme/theme.dart';
import 'package:turd/utils/helper/helper.dart';
import 'package:flutter/material.dart';


class InfoScreen extends StatelessWidget {
  const InfoScreen({super.key});

  @override
  Widget build(BuildContext context) {
    
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
      child: Container(
        width: MediaQuery.of(context).size.width * 0.9,
        child: Column(
          children: [
            Text(
              "About this app",
              style: TextStyle(
                fontSize: 25, 
                fontWeight: FontWeight.bold,
                color: HelperFunction.isDark(context)?Colors.black : Colors.white,
              ),
            ),
            const SizedBox(height: 10),
            Container(
              padding: EdgeInsets.only(bottom: 10),
              child: Text(
              "A product of BIN LABS", 
              style: TextStyle(
                fontSize: 14,
                color: HelperFunction.isDark(context)?Colors.black : Colors.white
              ),
              
              ),
            ),
            const SizedBox(height: 10),
            Container(
              width: double.infinity,
              child: Container(
                child: FilledButtonTheme(
                  
                  data: HelperFunction.isDark(context)?AppTheme.darkTheme.filledButtonTheme : AppTheme.lightTheme.filledButtonTheme,
                  child: FilledButton(
                    onPressed: () {
                      Navigator.pop(context);
                    },
                    child: Text("Exit"),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
