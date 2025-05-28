import 'package:flutter/material.dart';
import 'package:turd/pages/home_page.dart';

// class PredictionBox extends StatelessWidget {

//   const PredictionBox({super.key});

//   @override
//   Widget build(BuildContext context) {
//     return Card(
//       color: Colors.grey,
//       clipBehavior: Clip.hardEdge,
//       shape: RoundedRectangleBorder(
//         borderRadius: BorderRadius.circular(20),
//         side: BorderSide(color: Colors.pink),
//       ),
//       child: SizedBox(
//         width: MediaQuery.of(context).size.width * 0.9,
//         child: Column(
//           children: [

//             const SizedBox(height: 12),
//             Padding(
//               padding: const EdgeInsets.all(8.0),
//               child: Column(
//                 children: [
//                   Text("Prediction: ${BirdResult().prediction}"),
//                   Text(
//                     "Confidence: ${BirdResult().confidence.toStringAsFixed(2)}%",
//                   ),
//                 ],
//               ),
//             ),
//           ],
//         ),
//       ),
//     );
//   }
// }

class PredictionBox extends HomePageState {

  @override
  Widget build(BuildContext context) {
    return Card();
  }
}
